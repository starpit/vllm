// SPDX-License-Identifier: Apache-2.0
//! GGUF file loading, config extraction, and tensor name mapping.
//!
//! Thin wrapper around `candle_core::quantized::gguf_file::Content` that
//! provides typed metadata accessors and maps GGUF tensor names to the
//! HuggingFace convention used by the rest of the crate.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use tracing::info;

use candle_core::Device;
use candle_core::quantized::QTensor;
use candle_core::quantized::gguf_file::{Content, Value};

use crate::error::{ModelError, ModelResult};
use crate::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// GgufFile
// ---------------------------------------------------------------------------

/// Opened GGUF file with parsed header and lazy tensor reads.
pub struct GgufFile {
    content: Content,
    reader: BufReader<File>,
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
        Ok(Self { content, reader })
    }

    /// Read a quantized tensor by name.
    pub fn tensor(&mut self, name: &str, device: &Device) -> ModelResult<QTensor> {
        self.content
            .tensor(&mut self.reader, name, device)
            .map_err(|e| ModelError::Other(format!("failed to read tensor '{name}': {e}")))
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
            // Try u32 first, then upcast from smaller types.
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
        "gemma2" | "gemma" => "Gemma2ForCausalLM",
        "mistral" => "MistralForCausalLM",
        "phi3" | "phi" => "Phi3ForCausalLM",
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
    // Fallback: try general.vocab_size for some GGUF variants.
    if config.vocab_size.is_none() {
        config.vocab_size = gguf
            .get_metadata_u32("general.vocab_size")
            .map(|v| v as usize);
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

    // Compute head_dim if not explicitly stored.
    if let (Some(hidden), Some(heads)) = (config.hidden_size, config.num_attention_heads)
        && heads > 0
    {
        config.head_dim = Some(hidden / heads);
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
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_norm.weight" => "input_layernorm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            other => return format!("model.layers.{layer_num}.{other}"),
        };

        return format!("model.layers.{layer_num}.{hf_suffix}");
    }

    // Unknown — pass through unchanged.
    gguf_name.to_string()
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
    fn test_gguf_to_hf_name_unknown_passthrough() {
        assert_eq!(
            gguf_to_hf_name("some.unknown.tensor"),
            "some.unknown.tensor"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_unknown_layer_suffix() {
        // Unknown per-layer suffix passes through with model.layers prefix.
        assert_eq!(
            gguf_to_hf_name("blk.0.some_unknown.weight"),
            "model.layers.0.some_unknown.weight"
        );
    }
}
