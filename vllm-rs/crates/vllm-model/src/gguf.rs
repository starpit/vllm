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

    /// Get the length of an array metadata value.
    pub fn get_metadata_array_len(&self, key: &str) -> Option<usize> {
        use candle_core::quantized::gguf_file::Value;
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
        // inner_size = value_dim = num_v_heads * head_v_dim.
        // The attn_qkv output dim = 2*key_dim + 2*value_dim, where key_dim = num_k_heads * head_k_dim.
        // We can derive: head_k_dim = (qkvz_dim - 2*inner_size) / (2*num_k_heads)
        // But since we don't have qkvz_dim in metadata, use the embedding_length relationship:
        // For Qwen3.5-0.8B: embedding=1024, inner_size=2048, num_k_heads=16
        //   => qkvz_dim = 6144 from actual tensor, key_dim = (6144-4096)/2 = 1024, head_k_dim = 64
        // We store inner_size so the model loader can derive head_k_dim from the actual weight shape.
        if let Some(inner) = gguf.get_metadata_u32(&format!("{arch}.ssm.inner_size")) {
            config
                .extra
                .insert("ssm_inner_size".to_string(), serde_json::json!(inner));
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
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_norm.weight" => "input_layernorm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            other => return format!("model.layers.{layer_num}.{other}"),
        };

        return format!("model.layers.{layer_num}.{hf_suffix}");
    }

    // Unknown — pass through unchanged.
    gguf_name.to_string()
}

// ---------------------------------------------------------------------------
// Multimodal GGUF: mmproj detection and loading
// ---------------------------------------------------------------------------

/// Detect a sibling mmproj GGUF file next to a main GGUF file.
///
/// Looks for files matching `mmproj*.gguf` in the same directory as
/// `main_gguf_path`. Returns the first match, if any.
///
/// This matches the Python `detect_gguf_multimodal()` behavior.
pub fn detect_mmproj_gguf(main_gguf_path: &Path) -> Option<std::path::PathBuf> {
    let parent = main_gguf_path.parent()?;
    let entries = std::fs::read_dir(parent).ok()?;
    for entry in entries.flatten() {
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();
        if fname_str.starts_with("mmproj") && fname_str.ends_with(".gguf") {
            // Skip the main GGUF file itself if it happens to start with "mmproj".
            if entry.path() != main_gguf_path {
                return Some(entry.path());
            }
        }
    }
    None
}

/// Map an mmproj GGUF tensor name to the HuggingFace convention.
///
/// mmproj GGUFs use a compact naming scheme:
/// - `v.blk.{i}.attn_q.weight` → vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight
/// - `v.blk.{i}.ln1.weight` → ...layer_norm1.weight
/// - `v.blk.{i}.ffn_up.weight` → ...mlp.fc1.weight
/// - `v.patch_embd.weight` → ...embeddings.patch_embedding.weight
/// - `v.position_embd.weight` → ...embeddings.position_embedding.weight
/// - `v.post_ln.weight` → ...post_layernorm.weight
/// - `mm.0.weight` → multi_modal_projector.mm_input_projection_weight
/// - `mm.model_norm.weight` → multi_modal_projector.mm_soft_emb_norm.weight
fn mmproj_gguf_to_hf_name(gguf_name: &str) -> String {
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

/// Load an mmproj GGUF file and return dequantized weights as a `ModelWeights`.
///
/// All tensors are dequantized to f32 and mapped from GGUF mmproj naming
/// convention to HuggingFace naming convention.
pub fn load_mmproj_as_model_weights(
    mmproj_path: &Path,
    device: &Device,
) -> ModelResult<crate::weight::ModelWeights> {
    use std::collections::HashMap;

    info!("Loading mmproj GGUF from {}", mmproj_path.display());

    let mut gguf = GgufFile::open(mmproj_path)?;
    let tensor_names: Vec<String> = gguf.tensor_names().iter().map(|s| s.to_string()).collect();

    info!("mmproj GGUF has {} tensors", tensor_names.len());

    let mut tensors: HashMap<String, candle_core::Tensor> = HashMap::new();
    for name in &tensor_names {
        let qt = gguf.tensor(name, device)?;
        let t = qt
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize mmproj tensor '{name}': {e}")))?;
        let hf_name = mmproj_gguf_to_hf_name(name);
        tensors.insert(hf_name, t);
    }

    Ok(crate::weight::ModelWeights::from_tensors(tensors))
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
            // Default mm_tokens_per_image is num_patches (before any pooling).
            // Gemma3 uses 256 by default (after avg pool).
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
