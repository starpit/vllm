// SPDX-License-Identifier: Apache-2.0
//! GGUF file loading, config extraction, and tensor name mapping.
//!
//! Wraps `gguf_format::Content` (our vendored GGUF parser) and provides
//! typed metadata accessors and maps GGUF tensor names to the HuggingFace
//! convention used by the rest of the crate.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use tracing::info;

use crate::error::{ModelError, ModelResult};
use crate::gguf_format::{Content, Value};
use crate::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// GgufFile
// ---------------------------------------------------------------------------

/// Opened GGUF file with parsed header and lazy tensor reads.
pub struct GgufFile {
    pub(crate) content: Content,
}

impl GgufFile {
    /// Open and parse a GGUF file.
    pub fn open(path: impl AsRef<Path>) -> ModelResult<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| ModelError::Other(format!("failed to open {}: {e}", path.display())))?;
        let mut reader = BufReader::new(file);
        let content = Content::read(&mut reader).map_err(|e| {
            ModelError::Other(format!("failed to parse GGUF {}: {e}", path.display()))
        })?;
        Ok(Self { content })
    }

    /// List all tensor names in the GGUF file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.content
            .tensor_infos
            .keys()
            .map(|s| s.as_str())
            .collect()
    }

    /// Raw GGUF metadata.
    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.content.metadata
    }

    /// Get a string metadata value.
    pub fn get_metadata_string(&self, key: &str) -> Option<&str> {
        self.content
            .metadata
            .get(key)
            .and_then(|v| v.to_string().ok())
            .map(|s| s.as_str())
    }

    /// Get a u32 metadata value (auto-upcasts from smaller int types).
    pub fn get_metadata_u32(&self, key: &str) -> Option<u32> {
        self.content.metadata.get(key).and_then(|v| {
            if let Ok(val) = v.to_u32() {
                Some(val)
            } else if let Ok(val) = v.to_u64() {
                Some(val as u32)
            } else {
                None
            }
        })
    }

    /// Get an f32 metadata value.
    ///
    /// Also handles integer-stored values (some GGUF writers store floats
    /// like `rope.freq_base` as integers).
    pub fn get_metadata_f32(&self, key: &str) -> Option<f32> {
        self.content.metadata.get(key).and_then(|v| {
            if let Ok(val) = v.to_f32() {
                Some(val)
            } else if let Ok(val) = v.to_f64() {
                Some(val as f32)
            } else if let Ok(val) = v.to_u64() {
                Some(val as f32)
            } else if let Ok(val) = v.to_u32() {
                Some(val as f32)
            } else if let Ok(val) = v.to_i32() {
                Some(val as f32)
            } else {
                None
            }
        })
    }

    /// Get the length of an array metadata value.
    pub fn get_metadata_array_len(&self, key: &str) -> Option<usize> {
        self.content.metadata.get(key).and_then(|v| {
            if let Value::Array(arr) = v {
                Some(arr.len())
            } else {
                None
            }
        })
    }

    /// Total number of tensors in the file.
    pub fn num_tensors(&self) -> usize {
        self.content.tensor_infos.len()
    }
}

// ---------------------------------------------------------------------------
// GgufModelConfig — extract HfModelConfig from GGUF metadata
// ---------------------------------------------------------------------------

/// Extract model configuration from GGUF metadata.
///
/// GGUF files embed model hyperparameters in their metadata using keys like
/// `llama.block_count`, `llama.embedding_length`, etc. This function reads
/// those values and converts them to an `HfModelConfig` for compatibility
/// with the existing init code.
/// Extract the chat template (Jinja2) string from a GGUF file's
/// metadata. GGUF embeds the same `chat_template` field that HF's
/// `tokenizer_config.json` carries — see llama.cpp's
/// `convert_hf_to_gguf.py`. Returns `None` when the GGUF doesn't
/// ship a template (rare for chat-tuned models, common for base
/// completions models).
pub fn gguf_chat_template(gguf: &GgufFile) -> Option<String> {
    gguf.get_metadata_string("tokenizer.chat_template")
        .map(str::to_string)
}

/// Build a `tokenizers::Tokenizer` from GGUF metadata. Handles the
/// `gpt2` BPE family (Llama-3, Mistral-3-style models that use
/// `pre = "llama-bpe"` etc.) — vocab from
/// `tokenizer.ggml.tokens`, merges from `tokenizer.ggml.merges`,
/// special tokens from `tokenizer.ggml.{bos,eos,pad,unk}_token_id`.
///
/// Returns `Ok(None)` for GGUFs whose `tokenizer.ggml.model` is
/// neither `"gpt2"` nor `"llama"` (these would need
/// SentencePiece/Unigram support and a different builder); the
/// caller should fall back to a sibling `tokenizer.json` for those.
pub fn gguf_tokenizer(gguf: &GgufFile) -> ModelResult<Option<tokenizers::Tokenizer>> {
    use tokenizers::AddedToken;
    use tokenizers::SplitDelimiterBehavior;
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::PreTokenizerWrapper;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
    use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};

    let Some(model) = gguf.get_metadata_string("tokenizer.ggml.model") else {
        return Ok(None);
    };
    // Llama-3 / Mistral-3 / Qwen-2/3 / etc. all use `model = "gpt2"`
    // (byte-level BPE). `model = "llama"` is the older
    // SentencePiece-BPE used by Llama-2; structurally similar but
    // not byte-level — left as a follow-up.
    if model != "gpt2" {
        tracing::info!(
            "gguf_tokenizer: unsupported tokenizer model `{model}` (only `gpt2` BPE \
             is reconstructable today); caller should fall back to tokenizer.json"
        );
        return Ok(None);
    }

    // Vocab: array of strings indexed by token id.
    let tokens_val = gguf
        .metadata()
        .get("tokenizer.ggml.tokens")
        .ok_or_else(|| ModelError::Other("GGUF missing tokenizer.ggml.tokens".into()))?;
    let tokens = tokens_val
        .to_vec()
        .map_err(|e| ModelError::Other(format!("tokenizer.ggml.tokens: {e}")))?;
    let mut vocab_pairs: Vec<(String, u32)> = Vec::with_capacity(tokens.len());
    for (idx, t) in tokens.iter().enumerate() {
        let s = t
            .to_string()
            .map_err(|e| ModelError::Other(format!("tokenizer.ggml.tokens[{idx}]: {e}")))?;
        vocab_pairs.push((s.clone(), idx as u32));
    }
    let vocab: tokenizers::models::bpe::Vocab = vocab_pairs.into_iter().collect();

    // Merges: array of "left right" strings.
    let merges_val = gguf
        .metadata()
        .get("tokenizer.ggml.merges")
        .ok_or_else(|| ModelError::Other("GGUF missing tokenizer.ggml.merges".into()))?;
    let merges_raw = merges_val
        .to_vec()
        .map_err(|e| ModelError::Other(format!("tokenizer.ggml.merges: {e}")))?;
    let mut merges: Vec<(String, String)> = Vec::with_capacity(merges_raw.len());
    for (idx, m) in merges_raw.iter().enumerate() {
        let s = m
            .to_string()
            .map_err(|e| ModelError::Other(format!("tokenizer.ggml.merges[{idx}]: {e}")))?;
        let mut parts = s.splitn(2, ' ');
        let l = parts
            .next()
            .ok_or_else(|| ModelError::Other(format!("merge {idx}: empty")))?;
        let r = parts
            .next()
            .ok_or_else(|| ModelError::Other(format!("merge {idx}: missing right side `{s}`")))?;
        merges.push((l.to_string(), r.to_string()));
    }

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| ModelError::Other(format!("BPE build failed: {e}")))?;

    let mut tokenizer = tokenizers::Tokenizer::new(bpe);

    // Pre-tokenizer. `tokenizer.ggml.pre` names the family. Almost
    // every modern HF tokenizer is structured as:
    //   Sequence([Split(<arch_regex>, Isolated), ByteLevel(false, false, false)])
    // The arch differs only in the splitting regex — so the regex
    // table below is the single point of variance. ByteLevel always
    // runs with `add_prefix_space=false, trim_offsets=false,
    // use_regex=false` (the regex Split already did the splitting;
    // ByteLevel just maps bytes through the byte-to-unicode table).
    //
    // Wrong choice here is the #1 cause of gibberish output — picking
    // `ByteLevel::default()` (which has `add_prefix_space=true,
    // use_regex=true`) silently inserts a leading space on every
    // word, offsetting every token id and destroying inference.
    let pre = gguf
        .get_metadata_string("tokenizer.ggml.pre")
        .unwrap_or("default");
    let llama3_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    // Qwen2 / Qwen2.5 / Qwen3: same family as Llama-3 but `\p{N}`
    // instead of `\p{N}{1,3}` (digits split one-at-a-time).
    let qwen2_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    // GPT-2 base regex (also fits `default` and tokenizers whose
    // pre-tokenizer field is absent — the safest fallback that still
    // keeps `add_prefix_space=false` so we don't silently corrupt
    // tokenization).
    let gpt2_regex =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    let regex_str: &str = match pre {
        "llama-bpe" | "llama3" | "llama-v3" => llama3_regex,
        "qwen2" => qwen2_regex,
        _ => gpt2_regex,
    };
    let regex_split = Split::new(
        SplitPattern::Regex(regex_str.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| ModelError::Other(format!("pre-tokenizer regex compile ({pre}): {e}")))?;
    let bl = ByteLevelPre::new(false, false, false);
    let pre_tokenizer: PreTokenizerWrapper =
        PreSequence::new(vec![regex_split.into(), bl.into()]).into();
    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
    // Decoder: ByteLevel with `add_prefix_space=false, trim_offsets=false`
    // matches HF tokenizer.json on Llama-3/Qwen2/Qwen3/Mistral families.
    // The default ctor has `add_prefix_space=true` which strips a
    // leading space from every decoded token — visible as missing
    // spaces in user-facing output.
    tokenizer.with_decoder(Some(ByteLevelDecoder::new(false, false, false)));

    // Special tokens. GGUF marks token roles via
    // `tokenizer.ggml.token_type` (parallel to tokens array):
    //   1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED,
    //   5=UNUSED, 6=BYTE.
    // Anything that's CONTROL or USER_DEFINED needs to be registered
    // as a special added token so the tokenizer recognizes the
    // multi-byte form (e.g. `<|begin_of_text|>`) atomically rather
    // than BPE-encoding it. Without this, chat templates produce
    // nonsense input to the model.
    let mut added: Vec<AddedToken> = Vec::new();
    if let Some(types_val) = gguf.metadata().get("tokenizer.ggml.token_type")
        && let Ok(types) = types_val.to_vec()
    {
        for (idx, t) in types.iter().enumerate() {
            // GGUF token_type is typically I32 (signed) — try both.
            let kind = t
                .to_u32()
                .map(|x| x as i64)
                .or_else(|_| t.to_i32().map(|x| x as i64))
                .unwrap_or(1);
            if (kind == 3 || kind == 4)
                && let Some(token_val) = tokens.get(idx)
                && let Ok(s) = token_val.to_string()
            {
                added.push(AddedToken::from(s.clone(), true));
            }
        }
    }
    // Always add the explicit-id specials too (some GGUFs don't set
    // token_type for these or set to NORMAL).
    let token_str_for_id = |id: u32| -> Option<String> {
        let s = tokens.get(id as usize)?.to_string().ok()?.clone();
        Some(s)
    };
    for key in [
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.pad_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
    ] {
        if let Some(id) = gguf.get_metadata_u32(key)
            && let Some(content) = token_str_for_id(id)
        {
            added.push(AddedToken::from(content, true));
        }
    }
    if !added.is_empty() {
        tokenizer.add_special_tokens(&added);
    }

    Ok(Some(tokenizer))
}

pub fn gguf_model_config(gguf: &GgufFile) -> ModelResult<HfModelConfig> {
    // Determine the architecture prefix (e.g., "llama", "qwen2").
    let arch = gguf
        .get_metadata_string("general.architecture")
        .ok_or_else(|| ModelError::Other("GGUF missing general.architecture metadata".to_string()))?
        .to_string();

    let mut config = HfModelConfig {
        model_type: Some(arch.clone()),
        ..Default::default()
    };

    // Map GGUF architecture name to HF architecture class.
    let hf_arch = match arch.as_str() {
        "llama" => "LlamaForCausalLM",
        "qwen2" => "Qwen2ForCausalLM",
        "qwen3" => "Qwen3ForCausalLM",
        "qwen35" => "Qwen3NextForCausalLM",
        "gemma3" => "Gemma3ForCausalLM",
        "gemma2" | "gemma" => "Gemma2ForCausalLM",
        "mistral" => "MistralForCausalLM",
        "phi3" | "phi" => "Phi3ForCausalLM",
        "deepseek2" => "DeepseekV2ForCausalLM",
        other => other, // pass through as-is
    };
    config.architectures = vec![hf_arch.to_string()];

    // Read hyperparameters using the architecture prefix.
    config.num_hidden_layers = gguf
        .get_metadata_u32(&format!("{arch}.block_count"))
        .map(|v| v as usize);
    config.hidden_size = gguf
        .get_metadata_u32(&format!("{arch}.embedding_length"))
        .map(|v| v as usize);
    config.num_attention_heads = gguf
        .get_metadata_u32(&format!("{arch}.attention.head_count"))
        .map(|v| v as usize);
    config.num_key_value_heads = gguf
        .get_metadata_u32(&format!("{arch}.attention.head_count_kv"))
        .map(|v| v as usize);
    config.intermediate_size = gguf
        .get_metadata_u32(&format!("{arch}.feed_forward_length"))
        .map(|v| v as usize);
    config.vocab_size = gguf
        .get_metadata_u32(&format!("{arch}.vocab_size"))
        .map(|v| v as usize);
    // Fallback: try general.vocab_size, then tokenizer.ggml.tokens array length.
    if config.vocab_size.is_none() {
        config.vocab_size = gguf
            .get_metadata_u32("general.vocab_size")
            .map(|v| v as usize);
    }
    if config.vocab_size.is_none() {
        config.vocab_size = gguf.get_metadata_array_len("tokenizer.ggml.tokens");
    }
    config.max_position_embeddings = gguf
        .get_metadata_u32(&format!("{arch}.context_length"))
        .map(|v| v as usize);
    config.rope_theta = gguf
        .get_metadata_f32(&format!("{arch}.rope.freq_base"))
        .map(|v| v as f64);
    config.rms_norm_eps = gguf
        .get_metadata_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .map(|v| v as f64);

    // rope_scaling: parse if the GGUF records it, otherwise infer llama3 defaults
    // for Llama-3.x files (unsloth GGUFs omit the scaling keys entirely).
    // Keys follow llama.cpp convention: {arch}.rope.scaling.type and friends.
    let rope_scaling_type = gguf.get_metadata_string(&format!("{arch}.rope.scaling.type"));
    let has_scaling_keys = rope_scaling_type.is_some()
        || gguf
            .get_metadata_f32(&format!("{arch}.rope.scaling.factor"))
            .is_some();
    if has_scaling_keys {
        let rope_type = rope_scaling_type.unwrap_or("linear").to_string();
        let factor = gguf
            .get_metadata_f32(&format!("{arch}.rope.scaling.factor"))
            .unwrap_or(1.0) as f64;
        let mut scaling = serde_json::Map::new();
        scaling.insert(
            "rope_type".to_string(),
            serde_json::Value::String(rope_type.clone()),
        );
        scaling.insert("factor".to_string(), serde_json::Value::from(factor));
        if let Some(orig) =
            gguf.get_metadata_u32(&format!("{arch}.rope.scaling.original_context_length"))
        {
            scaling.insert(
                "original_max_position_embeddings".to_string(),
                serde_json::Value::from(orig as u64),
            );
        }
        if rope_type == "llama3" {
            if let Some(lo) = gguf.get_metadata_f32(&format!("{arch}.rope.scaling.low_freq_factor"))
            {
                scaling.insert(
                    "low_freq_factor".to_string(),
                    serde_json::Value::from(lo as f64),
                );
            }
            if let Some(hi) =
                gguf.get_metadata_f32(&format!("{arch}.rope.scaling.high_freq_factor"))
            {
                scaling.insert(
                    "high_freq_factor".to_string(),
                    serde_json::Value::from(hi as f64),
                );
            }
        }
        config.extra.insert(
            "rope_scaling".to_string(),
            serde_json::Value::Object(scaling),
        );
    } else if arch == "llama" {
        // Infer Llama-3.x llama3 rope_scaling when the GGUF omits the scaling keys.
        // Signature: rope_theta == 500000 (Llama 3 base) AND context_length > 8192 (extended).
        // Values match HF config for Llama-3.1 / 3.2 / 3.3 (all use identical defaults).
        let theta = config.rope_theta.unwrap_or(10000.0);
        let ctx = config.max_position_embeddings.unwrap_or(0);
        if (theta - 500000.0).abs() < 1.0 && ctx > 8192 {
            let mut scaling = serde_json::Map::new();
            scaling.insert(
                "rope_type".to_string(),
                serde_json::Value::String("llama3".to_string()),
            );
            scaling.insert("factor".to_string(), serde_json::Value::from(32.0));
            scaling.insert("low_freq_factor".to_string(), serde_json::Value::from(1.0));
            scaling.insert("high_freq_factor".to_string(), serde_json::Value::from(4.0));
            scaling.insert(
                "original_max_position_embeddings".to_string(),
                serde_json::Value::from(8192u64),
            );
            config.extra.insert(
                "rope_scaling".to_string(),
                serde_json::Value::Object(scaling),
            );
        }
    }

    // Compute head_dim: try arch-specific key_length first, fall back to hidden/heads.
    if let Some(key_len) = gguf.get_metadata_u32(&format!("{arch}.attention.key_length")) {
        config.head_dim = Some(key_len as usize);
    } else if let (Some(hidden), Some(heads)) = (config.hidden_size, config.num_attention_heads)
        && heads > 0
    {
        config.head_dim = Some(hidden / heads);
    }

    // Gemma3-specific metadata.
    if arch == "gemma3" {
        // Sliding window.
        if let Some(sw) = gguf.get_metadata_u32(&format!("{arch}.attention.sliding_window")) {
            config
                .extra
                .insert("sliding_window".to_string(), serde_json::json!(sw));
            // Default sliding_window_pattern = 6 when sliding_window is present.
            if !config.extra.contains_key("sliding_window_pattern") {
                config
                    .extra
                    .insert("sliding_window_pattern".to_string(), serde_json::json!(6));
            }
        }

        // Local RoPE theta (for sliding window layers).
        let local_freq = gguf
            .get_metadata_f32(&format!("{arch}.rope.local.freq_base"))
            .unwrap_or(10000.0);
        config.extra.insert(
            "rope_local_base_freq".to_string(),
            serde_json::json!(local_freq as f64),
        );

        // query_pre_attn_scalar defaults to head_dim.
        let head_dim = config.head_dim.unwrap_or(256);
        config.extra.insert(
            "query_pre_attn_scalar".to_string(),
            serde_json::json!(head_dim as f64),
        );
    }

    // Qwen3.5 (qwen35) specific metadata: SSM/hybrid attention fields.
    if arch == "qwen35" {
        if let Some(v) = gguf.get_metadata_u32(&format!("{arch}.ssm.conv_kernel")) {
            config
                .extra
                .insert("linear_conv_kernel_dim".to_string(), serde_json::json!(v));
        }
        if let Some(v) = gguf.get_metadata_u32(&format!("{arch}.ssm.state_size")) {
            config
                .extra
                .insert("linear_value_head_dim".to_string(), serde_json::json!(v));
        }
        if let Some(v) = gguf.get_metadata_u32(&format!("{arch}.ssm.group_count")) {
            config
                .extra
                .insert("linear_num_value_heads".to_string(), serde_json::json!(v));
            // linear_num_key_heads defaults to same as group_count.
            config
                .extra
                .insert("linear_num_key_heads".to_string(), serde_json::json!(v));
        }
        if let Some(v) = gguf.get_metadata_u32(&format!("{arch}.full_attention_interval")) {
            config
                .extra
                .insert("full_attention_interval".to_string(), serde_json::json!(v));
        }
        // Derive partial_rotary_factor from rope.dimension_count / head_dim.
        if let Some(rope_dim) = gguf.get_metadata_u32(&format!("{arch}.rope.dimension_count"))
            && let Some(hd) = config.head_dim
        {
            let factor = rope_dim as f64 / hd as f64;
            config.extra.insert(
                "partial_rotary_factor".to_string(),
                serde_json::json!(factor),
            );
        }
        // linear_num_key_heads from ssm.time_step_rank.
        if let Some(time_step_rank) = gguf.get_metadata_u32(&format!("{arch}.ssm.time_step_rank")) {
            config.extra.insert(
                "linear_num_key_heads".to_string(),
                serde_json::json!(time_step_rank),
            );
        }
        // linear_key_head_dim: derive from ssm.inner_size.
        if let Some(inner) = gguf.get_metadata_u32(&format!("{arch}.ssm.inner_size")) {
            config
                .extra
                .insert("ssm_inner_size".to_string(), serde_json::json!(inner));
        }
    }

    // DeepSeek V2/V3 specific metadata (MLA + MoE fields).
    if arch == "deepseek2" {
        // MLA attention dimensions
        if let Some(v) = gguf.get_metadata_u32(&format!("{arch}.attention.q_lora_rank")) {
            config
                .extra
                .insert("q_lora_rank".to_string(), serde_json::json!(v));
        }
        if let Some(v) = gguf.get_metadata_u32(&format!("{arch}.attention.kv_lora_rank")) {
            config
                .extra
                .insert("kv_lora_rank".to_string(), serde_json::json!(v));
        }
        // MLA head dimensions: try key_length_mla (new llama.cpp), then derive from
        // key_length + kv_lora_rank (older GGUF), then fall back to DeepSeek defaults.
        let rope_dim = gguf
            .get_metadata_u32(&format!("{arch}.rope.dimension_count"))
            .unwrap_or(64); // All DeepSeek V2/V3 variants use 64
        if let Some(key_len_mla) =
            gguf.get_metadata_u32(&format!("{arch}.attention.key_length_mla"))
        {
            // New format: key_length_mla = qk_nope + qk_rope
            let nope_dim = key_len_mla.saturating_sub(rope_dim);
            config
                .extra
                .insert("qk_nope_head_dim".to_string(), serde_json::json!(nope_dim));
            config
                .extra
                .insert("qk_rope_head_dim".to_string(), serde_json::json!(rope_dim));
        } else {
            // Older GGUF without key_length_mla: all DeepSeek V2/V3 variants
            // use qk_nope_head_dim=128, qk_rope_head_dim=64.
            config
                .extra
                .insert("qk_nope_head_dim".to_string(), serde_json::json!(128u32));
            config
                .extra
                .insert("qk_rope_head_dim".to_string(), serde_json::json!(rope_dim));
        }
        if let Some(v) = gguf.get_metadata_u32(&format!("{arch}.attention.value_length_mla")) {
            config
                .extra
                .insert("v_head_dim".to_string(), serde_json::json!(v));
        } else {
            // All DeepSeek V2/V3 variants use v_head_dim=128
            config
                .extra
                .insert("v_head_dim".to_string(), serde_json::json!(128u32));
        }
        // MoE fields
        if let Some(v) = gguf.get_metadata_u32(&format!("{arch}.expert_count")) {
            config
                .extra
                .insert("n_routed_experts".to_string(), serde_json::json!(v));
        }
        if let Some(v) = gguf.get_metadata_u32(&format!("{arch}.expert_used_count")) {
            config
                .extra
                .insert("num_experts_per_tok".to_string(), serde_json::json!(v));
        }
        if let Some(v) = gguf.get_metadata_u32(&format!("{arch}.expert_shared_count")) {
            config
                .extra
                .insert("n_shared_experts".to_string(), serde_json::json!(v));
        }
        if let Some(v) = gguf.get_metadata_u32(&format!("{arch}.expert_feed_forward_length")) {
            config
                .extra
                .insert("moe_intermediate_size".to_string(), serde_json::json!(v));
        }
        if let Some(v) = gguf.get_metadata_u32(&format!("{arch}.leading_dense_block_count")) {
            config
                .extra
                .insert("first_k_dense_replace".to_string(), serde_json::json!(v));
        }
        if let Some(v) = gguf.get_metadata_f32(&format!("{arch}.expert_weights_scale")) {
            config.extra.insert(
                "routed_scaling_factor".to_string(),
                serde_json::json!(v as f64),
            );
        }
    }

    // EOS token ID (stored in tokenizer.ggml.eos_token_id or general.eos_token_id).
    if let Some(eos) = gguf.get_metadata_u32("tokenizer.ggml.eos_token_id") {
        config
            .extra
            .insert("eos_token_id".to_string(), serde_json::json!(eos));
    }

    info!(
        "GGUF config: arch={}, hidden={:?}, heads={:?}, kv_heads={:?}, layers={:?}, \
         intermediate={:?}, vocab={:?}, max_pos={:?}, rope_theta={:?}, rms_eps={:?}, head_dim={:?}",
        arch,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.num_hidden_layers,
        config.intermediate_size,
        config.vocab_size,
        config.max_position_embeddings,
        config.rope_theta,
        config.rms_norm_eps,
        config.head_dim,
    );

    Ok(config)
}

// ---------------------------------------------------------------------------
// Tensor name mapping: GGUF (llama.cpp) → HuggingFace convention
// ---------------------------------------------------------------------------

/// Map a GGUF tensor name to the HuggingFace convention.
///
/// llama.cpp uses names like `blk.0.attn_q.weight` while HF uses
/// `model.layers.0.self_attn.q_proj.weight`. This mapping is needed so
/// that quantized models can reuse the same config/init code as safetensors
/// models.
pub fn gguf_to_hf_name(gguf_name: &str) -> String {
    // Global tensors.
    if gguf_name == "token_embd.weight" {
        return "model.embed_tokens.weight".to_string();
    }
    if gguf_name == "output_norm.weight" {
        return "model.norm.weight".to_string();
    }
    if gguf_name == "output.weight" {
        return "lm_head.weight".to_string();
    }

    // Per-layer tensors: blk.{i}.{suffix}
    if let Some(rest) = gguf_name.strip_prefix("blk.")
        && let Some(dot_pos) = rest.find('.')
    {
        let layer_num = &rest[..dot_pos];
        let suffix = &rest[dot_pos + 1..];

        let hf_suffix = match suffix {
            // Standard attention
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" | "attn_o.weight" => "self_attn.o_proj.weight",
            "attn_norm.weight" => "input_layernorm.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            // Biases (Qwen2/2.5 ships q/k/v biases; Llama / Mistral / etc. don't).
            // Without these arms the bias falls through to `model.layers.{N}.attn_q.bias`,
            // but the downstream loader looks for `model.layers.{N}.self_attn.q_proj.bias`,
            // and silently drops the bias — producing gibberish on biased archs.
            "attn_q.bias" => "self_attn.q_proj.bias",
            "attn_k.bias" => "self_attn.k_proj.bias",
            "attn_v.bias" => "self_attn.v_proj.bias",
            "attn_output.bias" | "attn_o.bias" => "self_attn.o_proj.bias",
            // MLA attention (DeepSeek V2/V3)
            "attn_q_a.weight" => "self_attn.q_a_proj.weight",
            "attn_q_a_norm.weight" => "self_attn.q_a_layernorm.weight",
            "attn_q_b.weight" => "self_attn.q_b_proj.weight",
            "attn_kv_a_mqa.weight" => "self_attn.kv_a_proj_with_mqa.weight",
            "attn_kv_a_norm.weight" => "self_attn.kv_a_layernorm.weight",
            "attn_kv_b.weight" => "self_attn.kv_b_proj.weight",
            // Standard MLP
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            // MoE router gate
            "ffn_gate_inp.weight" => "mlp.gate.weight",
            // Fused 3D expert weights (kept as fused_ prefix for loader to handle)
            "ffn_gate_exps.weight" => "mlp.experts.fused_gate_exps.weight",
            "ffn_up_exps.weight" => "mlp.experts.fused_up_exps.weight",
            "ffn_down_exps.weight" => "mlp.experts.fused_down_exps.weight",
            // Shared experts (DeepSeek V2/V3)
            "ffn_gate_shexp.weight" => "mlp.shared_experts.gate_proj.weight",
            "ffn_up_shexp.weight" => "mlp.shared_experts.up_proj.weight",
            "ffn_down_shexp.weight" => "mlp.shared_experts.down_proj.weight",
            // DeepSeek V3 score correction bias
            "exp_probs_b.bias" => "mlp.gate.e_score_correction_bias",
            other => return format!("model.layers.{layer_num}.{other}"),
        };

        return format!("model.layers.{layer_num}.{hf_suffix}");
    }

    // Unknown — pass through unchanged.
    gguf_name.to_string()
}

// ---------------------------------------------------------------------------
// Multimodal GGUF: mmproj detection
// ---------------------------------------------------------------------------

/// Detect a sibling mmproj GGUF file next to a main GGUF file.
///
/// Looks for files matching `mmproj*.gguf` in the same directory as
/// `main_gguf_path`. Returns the first match, if any.
pub fn detect_mmproj_gguf(main_gguf_path: &Path) -> Option<std::path::PathBuf> {
    let parent = main_gguf_path.parent()?;
    let entries = std::fs::read_dir(parent).ok()?;
    for entry in entries.flatten() {
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();
        if fname_str.starts_with("mmproj")
            && fname_str.ends_with(".gguf")
            && entry.path() != main_gguf_path
        {
            return Some(entry.path());
        }
    }
    None
}

/// Map an mmproj GGUF tensor name to the HuggingFace convention.
pub fn mmproj_gguf_to_hf_name(gguf_name: &str) -> String {
    // Projector tensors.
    if gguf_name == "mm.0.weight" {
        return "multi_modal_projector.mm_input_projection_weight".to_string();
    }
    if gguf_name == "mm.model_norm.weight" {
        return "multi_modal_projector.mm_soft_emb_norm.weight".to_string();
    }

    // Vision global tensors.
    if gguf_name == "v.patch_embd.weight" {
        return "vision_tower.vision_model.embeddings.patch_embedding.weight".to_string();
    }
    if gguf_name == "v.patch_embd.bias" {
        return "vision_tower.vision_model.embeddings.patch_embedding.bias".to_string();
    }
    if gguf_name == "v.position_embd.weight" {
        return "vision_tower.vision_model.embeddings.position_embedding.weight".to_string();
    }
    if gguf_name == "v.post_ln.weight" {
        return "vision_tower.vision_model.post_layernorm.weight".to_string();
    }
    if gguf_name == "v.post_ln.bias" {
        return "vision_tower.vision_model.post_layernorm.bias".to_string();
    }

    // Vision encoder per-layer tensors: v.blk.{i}.{suffix}
    if let Some(rest) = gguf_name.strip_prefix("v.blk.")
        && let Some(dot_pos) = rest.find('.')
    {
        let layer_num = &rest[..dot_pos];
        let suffix = &rest[dot_pos + 1..];
        let prefix = format!("vision_tower.vision_model.encoder.layers.{layer_num}");

        let hf_suffix = match suffix {
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_q.bias" => "self_attn.q_proj.bias",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_k.bias" => "self_attn.k_proj.bias",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_v.bias" => "self_attn.v_proj.bias",
            "attn_output.weight" => "self_attn.out_proj.weight",
            "attn_output.bias" => "self_attn.out_proj.bias",
            "ln1.weight" => "layer_norm1.weight",
            "ln1.bias" => "layer_norm1.bias",
            "ln2.weight" => "layer_norm2.weight",
            "ln2.bias" => "layer_norm2.bias",
            "ffn_up.weight" => "mlp.fc1.weight",
            "ffn_up.bias" => "mlp.fc1.bias",
            "ffn_down.weight" => "mlp.fc2.weight",
            "ffn_down.bias" => "mlp.fc2.bias",
            other => return format!("{prefix}.{other}"),
        };

        return format!("{prefix}.{hf_suffix}");
    }

    // Unknown — pass through unchanged.
    gguf_name.to_string()
}

/// Extract vision config from mmproj GGUF metadata and populate the
/// HfModelConfig extra fields needed for multimodal loading.
pub fn extract_mmproj_vision_config(
    mmproj_path: &Path,
    config: &mut HfModelConfig,
) -> ModelResult<()> {
    let gguf = GgufFile::open(mmproj_path)?;
    let meta = gguf.metadata();

    // Build vision_config JSON from mmproj metadata.
    let mut vision = serde_json::Map::new();

    // Try clip.vision.* keys (llama.cpp mmproj convention).
    if let Some(v) = meta
        .get("clip.vision.image_size")
        .and_then(|v| v.to_u32().ok())
    {
        vision.insert("image_size".to_string(), serde_json::json!(v));
    }
    if let Some(v) = meta
        .get("clip.vision.patch_size")
        .and_then(|v| v.to_u32().ok())
    {
        vision.insert("patch_size".to_string(), serde_json::json!(v));
    }
    if let Some(v) = meta
        .get("clip.vision.embedding_length")
        .and_then(|v| v.to_u32().ok())
    {
        vision.insert("hidden_size".to_string(), serde_json::json!(v));
    }
    if let Some(v) = meta
        .get("clip.vision.feed_forward_length")
        .and_then(|v| v.to_u32().ok())
    {
        vision.insert("intermediate_size".to_string(), serde_json::json!(v));
    }
    if let Some(v) = meta
        .get("clip.vision.block_count")
        .and_then(|v| v.to_u32().ok())
    {
        vision.insert("num_hidden_layers".to_string(), serde_json::json!(v));
    }
    if let Some(v) = meta
        .get("clip.vision.head_count")
        .and_then(|v| v.to_u32().ok())
    {
        vision.insert("num_attention_heads".to_string(), serde_json::json!(v));
    }

    // layer_norm_eps: try clip.vision.attention.layer_norm_epsilon.
    if let Some(v) = meta
        .get("clip.vision.attention.layer_norm_epsilon")
        .and_then(|v| v.to_f32().ok())
    {
        vision.insert("layer_norm_eps".to_string(), serde_json::json!(v as f64));
    }

    if !vision.is_empty() {
        config.extra.insert(
            "vision_config".to_string(),
            serde_json::Value::Object(vision),
        );

        // mm_tokens_per_image: compute from image_size and patch_size if available.
        if let (Some(img_size), Some(patch_size)) = (
            config
                .extra
                .get("vision_config")
                .and_then(|v| v.get("image_size"))
                .and_then(|v| v.as_u64()),
            config
                .extra
                .get("vision_config")
                .and_then(|v| v.get("patch_size"))
                .and_then(|v| v.as_u64()),
        ) {
            let patches_per_side = img_size / patch_size;
            let num_patches = patches_per_side * patches_per_side;
            if !config.extra.contains_key("mm_tokens_per_image") {
                config.extra.insert(
                    "mm_tokens_per_image".to_string(),
                    serde_json::json!(num_patches),
                );
            }
        }

        info!(
            "Extracted vision config from mmproj: {:?}",
            config.extra.get("vision_config")
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Inspect tensor shapes in a real GGUF file (run manually).
    #[test]
    #[ignore]
    fn inspect_gguf_shapes() {
        let home = std::env::var("HOME").unwrap();
        let path = format!(
            "{home}/.cache/huggingface/hub/models--unsloth--Llama-3.1-8B-Instruct-GGUF/snapshots/600b0020115fd6b17f0752848fe7b5a1be686bcd/Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        );
        if !std::path::Path::new(&path).exists() {
            eprintln!("GGUF file not found, skipping");
            return;
        }
        let gguf = GgufFile::open(&path).unwrap();
        let names = [
            "token_embd.weight",
            "output.weight",
            "output_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_norm.weight",
        ];
        for name in names {
            if let Some(info) = gguf.content.tensor_infos.get(name) {
                println!(
                    "{name}: shape={:?}, dtype={:?}",
                    info.shape.dims(),
                    info.ggml_dtype
                );
            } else {
                println!("{name}: NOT FOUND");
            }
        }
    }

    #[test]
    fn test_gguf_to_hf_name_embedding() {
        assert_eq!(
            gguf_to_hf_name("token_embd.weight"),
            "model.embed_tokens.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_output_norm() {
        assert_eq!(gguf_to_hf_name("output_norm.weight"), "model.norm.weight");
    }

    #[test]
    fn test_gguf_to_hf_name_lm_head() {
        assert_eq!(gguf_to_hf_name("output.weight"), "lm_head.weight");
    }

    #[test]
    fn test_gguf_to_hf_name_attention() {
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_q.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.5.attn_k.weight"),
            "model.layers.5.self_attn.k_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.31.attn_v.weight"),
            "model.layers.31.self_attn.v_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_output.weight"),
            "model.layers.0.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_norms() {
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_norm.weight"),
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ffn_norm.weight"),
            "model.layers.0.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_mlp() {
        assert_eq!(
            gguf_to_hf_name("blk.0.ffn_gate.weight"),
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ffn_up.weight"),
            "model.layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ffn_down.weight"),
            "model.layers.0.mlp.down_proj.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_deepseek_mla() {
        // MLA attention
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_q_a.weight"),
            "model.layers.0.self_attn.q_a_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_q_a_norm.weight"),
            "model.layers.0.self_attn.q_a_layernorm.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_q_b.weight"),
            "model.layers.0.self_attn.q_b_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_kv_a_mqa.weight"),
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_kv_a_norm.weight"),
            "model.layers.0.self_attn.kv_a_layernorm.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_kv_b.weight"),
            "model.layers.0.self_attn.kv_b_proj.weight"
        );
        // DeepSeek uses attn_o (not attn_output)
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_o.weight"),
            "model.layers.0.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_deepseek_moe() {
        // Router gate
        assert_eq!(
            gguf_to_hf_name("blk.5.ffn_gate_inp.weight"),
            "model.layers.5.mlp.gate.weight"
        );
        // Fused expert weights
        assert_eq!(
            gguf_to_hf_name("blk.5.ffn_gate_exps.weight"),
            "model.layers.5.mlp.experts.fused_gate_exps.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.5.ffn_up_exps.weight"),
            "model.layers.5.mlp.experts.fused_up_exps.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.5.ffn_down_exps.weight"),
            "model.layers.5.mlp.experts.fused_down_exps.weight"
        );
        // Shared experts
        assert_eq!(
            gguf_to_hf_name("blk.5.ffn_gate_shexp.weight"),
            "model.layers.5.mlp.shared_experts.gate_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.5.ffn_up_shexp.weight"),
            "model.layers.5.mlp.shared_experts.up_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.5.ffn_down_shexp.weight"),
            "model.layers.5.mlp.shared_experts.down_proj.weight"
        );
        // Score correction bias
        assert_eq!(
            gguf_to_hf_name("blk.5.exp_probs_b.bias"),
            "model.layers.5.mlp.gate.e_score_correction_bias"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_unknown_passthrough() {
        assert_eq!(
            gguf_to_hf_name("some.unknown.tensor"),
            "some.unknown.tensor"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_unknown_layer_suffix() {
        assert_eq!(
            gguf_to_hf_name("blk.0.some_unknown.weight"),
            "model.layers.0.some_unknown.weight"
        );
    }

    // --- mmproj name mapping tests ---

    #[test]
    fn test_mmproj_projector_names() {
        assert_eq!(
            mmproj_gguf_to_hf_name("mm.0.weight"),
            "multi_modal_projector.mm_input_projection_weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("mm.model_norm.weight"),
            "multi_modal_projector.mm_soft_emb_norm.weight"
        );
    }

    #[test]
    fn test_mmproj_vision_global_names() {
        assert_eq!(
            mmproj_gguf_to_hf_name("v.patch_embd.weight"),
            "vision_tower.vision_model.embeddings.patch_embedding.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.position_embd.weight"),
            "vision_tower.vision_model.embeddings.position_embedding.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.post_ln.weight"),
            "vision_tower.vision_model.post_layernorm.weight"
        );
    }

    #[test]
    fn test_mmproj_vision_layer_names() {
        assert_eq!(
            mmproj_gguf_to_hf_name("v.blk.0.attn_q.weight"),
            "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.blk.5.attn_output.weight"),
            "vision_tower.vision_model.encoder.layers.5.self_attn.out_proj.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.blk.0.ln1.weight"),
            "vision_tower.vision_model.encoder.layers.0.layer_norm1.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.blk.0.ln2.weight"),
            "vision_tower.vision_model.encoder.layers.0.layer_norm2.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.blk.0.ffn_up.weight"),
            "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.blk.0.ffn_down.weight"),
            "vision_tower.vision_model.encoder.layers.0.mlp.fc2.weight"
        );
    }

    #[test]
    fn test_mmproj_unknown_passthrough() {
        assert_eq!(
            mmproj_gguf_to_hf_name("some.random.tensor"),
            "some.random.tensor"
        );
    }

    #[test]
    fn test_detect_mmproj_no_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let main = dir.path().join("model.gguf");
        std::fs::write(&main, b"dummy").unwrap();
        assert!(detect_mmproj_gguf(&main).is_none());
    }

    #[test]
    fn test_detect_mmproj_with_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let main = dir.path().join("model.gguf");
        let mmproj = dir.path().join("mmproj-model-f16.gguf");
        std::fs::write(&main, b"dummy").unwrap();
        std::fs::write(&mmproj, b"dummy").unwrap();
        let found = detect_mmproj_gguf(&main);
        assert!(found.is_some());
        assert_eq!(found.unwrap(), mmproj);
    }
}
