// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Gemma 3 multimodal (vision-language) model for Candle.
//!
//! Implements `Gemma3ForConditionalGeneration` which wraps:
//! - `SiglipVisionModel` — vision encoder
//! - `Gemma3MultiModalProjector` — AvgPool2d → GemmaRMSNorm → matmul(projection_weight)
//! - `Gemma3ForCausalLM` — existing text-only language model
//!
//! The VLM model processes images through the vision encoder, projects them
//! into the LLM's embedding space, merges with text embeddings at placeholder
//! positions, and runs the language model backbone.
//!
//! Weight prefix mapping (HF checkpoint → code):
//! - `vision_tower.vision_model.*` → vision encoder
//! - `multi_modal_projector.*` → projector
//! - `language_model.model.*` → text backbone

use candle_core::{DType, Device, Module, Tensor};

use vllm_common::multimodal::MultimodalData;
use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{GemmaRmsNorm, Linear};
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::gemma3::{Gemma3Config, Gemma3ForCausalLM, Gemma3Model};
use crate::siglip::{SiglipVisionConfig, SiglipVisionModel};

// ---------------------------------------------------------------------------
// Gemma3VisionConfig — parsed from HfModelConfig
// ---------------------------------------------------------------------------

/// Configuration for the multimodal Gemma 3 model.
pub struct Gemma3VisionModelConfig {
    /// Vision encoder config.
    vision_config: SiglipVisionConfig,
    /// Text model config.
    text_config: Gemma3Config,
    /// Hidden size of the vision encoder.
    #[allow(dead_code)]
    vision_hidden_size: usize,
    /// Projection dimension (usually same as text hidden size).
    #[allow(dead_code)]
    projection_dim: usize,
    /// Token ID for image placeholders.
    image_token_index: u32,
    /// Number of image tokens per image in the token sequence.
    mm_tokens_per_image: usize,
    /// Number of patches per side: image_size / patch_size.
    patches_per_image: usize,
    /// AvgPool2d kernel size: patches_per_image / tokens_per_side.
    pool_kernel_size: usize,
}

impl Gemma3VisionModelConfig {
    fn from_hf_config(config: &HfModelConfig) -> ModelResult<Self> {
        // Parse vision_config sub-object.
        let vision_json = config
            .extra
            .get("vision_config")
            .ok_or_else(|| ModelError::Other("missing vision_config".into()))?;
        let vision_config = SiglipVisionConfig::from_json(vision_json)?;

        // Parse text_config sub-object, or fall back to top-level fields.
        let text_config = if let Some(text_json) = config.extra.get("text_config") {
            let text_hf: HfModelConfig = serde_json::from_value(text_json.clone())
                .map_err(|e| ModelError::Other(format!("failed to parse text_config: {e}")))?;
            Gemma3Config::from_hf_config(&text_hf)?
        } else {
            Gemma3Config::from_hf_config(config)?
        };

        let vision_hidden_size = vision_config.hidden_size;
        let projection_dim = text_config.hidden_size;

        let image_token_index = config
            .extra
            .get("image_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(255999) as u32;

        let mm_tokens_per_image = config
            .extra
            .get("mm_tokens_per_image")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(vision_config.num_patches());

        let patches_per_image = vision_config.image_size / vision_config.patch_size;
        let tokens_per_side = (mm_tokens_per_image as f64).sqrt() as usize;
        let pool_kernel_size = patches_per_image / tokens_per_side;

        Ok(Self {
            vision_config,
            text_config,
            vision_hidden_size,
            projection_dim,
            image_token_index,
            mm_tokens_per_image,
            patches_per_image,
            pool_kernel_size,
        })
    }
}

// ---------------------------------------------------------------------------
// Gemma3MultiModalProjector
// ---------------------------------------------------------------------------

/// Projects vision encoder hidden states into the LLM's embedding space.
///
/// Architecture: AvgPool2d → GemmaRMSNorm → matmul(projection_weight)
///
/// Port of: `vllm/model_executor/models/gemma3_mm.py::Gemma3MultiModalProjector`
struct Gemma3MultiModalProjector {
    /// Raw projection weight `[vision_hidden, text_hidden]`.
    mm_input_projection_weight: Tensor,
    /// GemmaRMSNorm applied after pooling.
    mm_soft_emb_norm: GemmaRmsNorm,
    /// Number of vision patches per side (image_size / patch_size).
    patches_per_image: usize,
    /// AvgPool2d kernel/stride size.
    kernel_size: usize,
}

impl Gemma3MultiModalProjector {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        patches_per_image: usize,
        kernel_size: usize,
        norm_eps: f64,
        dtype: DType,
    ) -> ModelResult<Self> {
        let mm_input_projection_weight =
            weights.get_cast(&format!("{prefix}.mm_input_projection_weight"), dtype)?;
        let mm_soft_emb_norm = GemmaRmsNorm::load(
            weights,
            &format!("{prefix}.mm_soft_emb_norm"),
            norm_eps,
            dtype,
        )?;

        Ok(Self {
            mm_input_projection_weight,
            mm_soft_emb_norm,
            patches_per_image,
            kernel_size,
        })
    }

    /// Forward: vision_outputs `[B, num_patches, vision_hidden]` → `[B, pooled_tokens, text_hidden]`
    fn forward(&self, vision_outputs: &Tensor) -> ModelResult<Tensor> {
        let (batch, _num_patches, seq_length) =
            vision_outputs.dims3().map_err(ModelError::Candle)?;

        // Transpose: [B, num_patches, vision_hidden] → [B, vision_hidden, num_patches]
        let x = vision_outputs.transpose(1, 2).map_err(ModelError::Candle)?;

        // Reshape to 2D grid: [B, vision_hidden, grid, grid]
        let grid = self.patches_per_image;
        let x = x
            .reshape((batch, seq_length, grid, grid))
            .map_err(ModelError::Candle)?;

        // AvgPool2d with kernel_size stride — reshape blocks and mean.
        let k = self.kernel_size;
        let out_grid = grid / k;
        // [B, C, grid, grid] → [B, C, out_grid, k, out_grid, k]
        let x = x
            .reshape((batch, seq_length, out_grid, k, out_grid, k))
            .map_err(ModelError::Candle)?;
        // Mean over the two kernel dims (3 and 5).
        let x = x
            .mean_keepdim(5)
            .map_err(ModelError::Candle)?
            .mean_keepdim(3)
            .map_err(ModelError::Candle)?;
        // Squeeze: [B, C, out_grid, 1, out_grid, 1] → [B, C, out_grid, out_grid]
        let x = x
            .reshape((batch, seq_length, out_grid, out_grid))
            .map_err(ModelError::Candle)?;

        // Flatten spatial: [B, C, out_grid, out_grid] → [B, C, pooled_tokens]
        let pooled_tokens = out_grid * out_grid;
        let x = x
            .reshape((batch, seq_length, pooled_tokens))
            .map_err(ModelError::Candle)?;

        // Transpose: [B, C, pooled_tokens] → [B, pooled_tokens, C]
        let x = x.transpose(1, 2).map_err(ModelError::Candle)?;

        // GemmaRMSNorm (operates on last dim).
        let x = self
            .mm_soft_emb_norm
            .forward(&x)
            .map_err(ModelError::Candle)?;

        // Matmul with projection weight: flatten to 2D for candle compatibility.
        // [B, pooled_tokens, vision_hidden] → [B*pooled_tokens, vision_hidden]
        let (b, t, _c) = x.dims3().map_err(ModelError::Candle)?;
        let x = x.reshape((b * t, ())).map_err(ModelError::Candle)?;
        // [B*pooled_tokens, vision_hidden] @ [vision_hidden, text_hidden] → [B*pooled_tokens, text_hidden]
        let x = x
            .matmul(&self.mm_input_projection_weight)
            .map_err(ModelError::Candle)?;
        // Reshape back: [B, pooled_tokens, text_hidden]
        let text_hidden = x.dim(1).map_err(ModelError::Candle)?;
        x.reshape((b, t, text_hidden)).map_err(ModelError::Candle)
    }

    #[cfg(test)]
    fn zeros(
        vision_hidden_size: usize,
        projection_dim: usize,
        patches_per_image: usize,
        kernel_size: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let mm_input_projection_weight =
            Tensor::zeros((vision_hidden_size, projection_dim), dtype, device)
                .map_err(ModelError::Candle)?;
        let mm_soft_emb_norm = GemmaRmsNorm::zeros(vision_hidden_size, 1e-6, dtype, device)?;
        Ok(Self {
            mm_input_projection_weight,
            mm_soft_emb_norm,
            patches_per_image,
            kernel_size,
        })
    }
}

// ---------------------------------------------------------------------------
// Gemma3ForConditionalGeneration
// ---------------------------------------------------------------------------

/// Gemma 3 multimodal model: vision encoder + projector + text LM.
pub struct Gemma3ForConditionalGeneration {
    vision_tower: SiglipVisionModel,
    multi_modal_projector: Gemma3MultiModalProjector,
    language_model: Gemma3ForCausalLM,
    #[allow(dead_code)]
    image_token_index: u32,
    #[allow(dead_code)]
    mm_tokens_per_image: usize,
    /// Stashed multimodal data for the next forward pass.
    stashed_mm_data: Option<MultimodalData>,
    dtype: DType,
}

impl Gemma3ForConditionalGeneration {
    /// Load from weights.
    pub fn load(
        weights: &ModelWeights,
        config: &Gemma3VisionModelConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        // Vision tower — weights prefixed with "vision_tower.vision_model."
        let vision_tower = SiglipVisionModel::load(
            weights,
            "vision_tower.vision_model",
            &config.vision_config,
            dtype,
        )?;

        // Projector — weights prefixed with "multi_modal_projector."
        let multi_modal_projector = Gemma3MultiModalProjector::load(
            weights,
            "multi_modal_projector",
            config.patches_per_image,
            config.pool_kernel_size,
            config.vision_config.layer_norm_eps,
            dtype,
        )?;

        // Language model — weights prefixed with "language_model.model."
        // Build directly with prefix instead of using Gemma3ForCausalLM::load
        // (which hardcodes "model" prefix for text-only use).
        let model = Gemma3Model::load(
            weights,
            "language_model.model",
            &config.text_config,
            dtype,
            device,
            0,
            1,
        )?;
        let lm_head = Linear::new(model.embed_tokens.weight().clone(), None);
        let language_model = Gemma3ForCausalLM {
            model,
            lm_head,
            final_logit_softcapping: config.text_config.final_logit_softcapping,
        };

        Ok(Self {
            vision_tower,
            multi_modal_projector,
            language_model,
            image_token_index: config.image_token_index,
            mm_tokens_per_image: config.mm_tokens_per_image,
            stashed_mm_data: None,
            dtype,
        })
    }

    /// Encode images and merge with text embeddings.
    fn merge_vision_embeddings(
        &self,
        input_ids: &Tensor,
        mm_data: &MultimodalData,
    ) -> ModelResult<Tensor> {
        // Get text embeddings.
        let text_embeds = self.language_model.model.embed(input_ids)?;
        let device = text_embeds.device().clone();

        if mm_data.images.is_empty() {
            return Ok(text_embeds);
        }

        // Build pixel tensor [N_images, 3, H, W] from ImageData.
        let pixel_tensors: Vec<Tensor> = mm_data
            .images
            .iter()
            .map(|img| {
                Tensor::from_vec(img.pixels.clone(), (1, 3, img.height, img.width), &device)
                    .map_err(ModelError::Candle)?
                    .to_dtype(self.dtype)
                    .map_err(ModelError::Candle)
            })
            .collect::<ModelResult<Vec<_>>>()?;
        let pixel_values = Tensor::cat(&pixel_tensors, 0).map_err(ModelError::Candle)?;

        // Encode: [N, 3, H, W] → [N, num_patches, vision_hidden]
        let vision_outputs = self.vision_tower.forward(&pixel_values)?;

        // Project: [N, num_patches, vision_hidden] → [N, pooled_tokens, text_hidden]
        let projected = self.multi_modal_projector.forward(&vision_outputs)?;
        let n_images = projected.dim(0).map_err(ModelError::Candle)?;

        // Scatter image embeddings into text embeddings at placeholder positions.
        let mut merged = text_embeds;
        for (img_idx, placeholder) in mm_data.image_placeholders.iter().enumerate() {
            if img_idx >= n_images {
                break;
            }
            let image_embeds = projected.get(img_idx).map_err(ModelError::Candle)?; // [pooled_tokens, hidden]
            let n_embed_tokens = image_embeds.dim(0).map_err(ModelError::Candle)?;

            // Replace tokens at offset..offset+length with image embeddings.
            let num_tokens = merged.dim(0).map_err(ModelError::Candle)?;
            let end = (placeholder.offset + placeholder.length).min(num_tokens);
            let actual_len = end.saturating_sub(placeholder.offset);
            if actual_len == 0 {
                continue;
            }

            // Trim image embeds if needed.
            let use_len = actual_len.min(n_embed_tokens);
            let image_embeds = image_embeds
                .narrow(0, 0, use_len)
                .map_err(ModelError::Candle)?;

            // Build merged = [before, image_embeds, after].
            let mut parts = Vec::new();
            if placeholder.offset > 0 {
                parts.push(
                    merged
                        .narrow(0, 0, placeholder.offset)
                        .map_err(ModelError::Candle)?,
                );
            }
            parts.push(image_embeds);
            if end < num_tokens {
                parts.push(
                    merged
                        .narrow(0, end, num_tokens - end)
                        .map_err(ModelError::Candle)?,
                );
            }
            merged = Tensor::cat(&parts, 0).map_err(ModelError::Candle)?;
        }

        Ok(merged)
    }
}

impl crate::Model for Gemma3ForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        if let Some(mm_data) = &self.stashed_mm_data {
            // Multimodal forward: merge vision + text, use forward_embeds.
            let merged_embeds = self.merge_vision_embeddings(input_ids, mm_data)?;
            self.language_model
                .forward_embeds(&merged_embeds, positions, kv_cache)
        } else {
            // Text-only forward.
            self.language_model.forward(input_ids, positions, kv_cache)
        }
    }

    fn forward_embeds(
        &self,
        inputs_embeds: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        self.language_model
            .forward_embeds(inputs_embeds, positions, kv_cache)
    }

    fn set_mm_data(&mut self, mm_data: Option<MultimodalData>) {
        self.stashed_mm_data = mm_data;
    }

    fn num_layers(&self) -> usize {
        self.language_model.num_layers()
    }
}

/// Factory function for the model registry.
pub fn create_gemma3_mm(
    weights: &ModelWeights,
    config: &HfModelConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let mm_config = Gemma3VisionModelConfig::from_hf_config(config)?;
    let model = Gemma3ForConditionalGeneration::load(weights, &mm_config, dtype, device)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolve the HF cache snapshot directory for a given model ID.
    /// Returns `None` if the model is not cached locally.
    fn resolve_hf_cache_dir(model_id: &str) -> Option<std::path::PathBuf> {
        let hf_home = std::env::var("HF_HOME")
            .unwrap_or_else(|_| format!("{}/.cache/huggingface", std::env::var("HOME").unwrap()));
        let dir_name = format!("models--{}", model_id.replace('/', "--"));
        let models_dir = std::path::PathBuf::from(hf_home).join("hub").join(dir_name);
        if !models_dir.exists() {
            return None;
        }
        // Find the first snapshot directory.
        let snapshots = models_dir.join("snapshots");
        std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.is_dir())
    }

    /// Collect all tensor names from a model directory (single or sharded safetensors).
    fn collect_tensor_names(model_dir: &std::path::Path) -> Vec<String> {
        use vllm_model::weight::{SafeTensorsFile, SafeTensorsIndex};

        let index_path = model_dir.join("model.safetensors.index.json");
        let single_path = model_dir.join("model.safetensors");

        if index_path.exists() {
            let index = SafeTensorsIndex::from_file(&index_path).unwrap();
            index.weight_map.keys().cloned().collect()
        } else if single_path.exists() {
            let file = SafeTensorsFile::open(&single_path).unwrap();
            file.tensor_names().unwrap()
        } else {
            panic!("No safetensors files in {}", model_dir.display());
        }
    }

    /// Validate that the weight prefixes our VLM code expects actually exist
    /// in the real HF checkpoint. This catches prefix mismatches without needing
    /// to load multi-GB weights.
    #[test]
    fn test_weight_names_match_hf_checkpoint_google_gemma3_4b_it() {
        let model_dir = match resolve_hf_cache_dir("google/gemma-3-4b-it") {
            Some(d) => d,
            None => {
                eprintln!("SKIP: google/gemma-3-4b-it not cached locally");
                return;
            }
        };

        let names = collect_tensor_names(&model_dir);
        let name_set: std::collections::HashSet<&str> = names.iter().map(|s| s.as_str()).collect();

        // --- Vision tower (prefix: "vision_tower.vision_model") ---
        assert!(
            name_set.contains("vision_tower.vision_model.embeddings.patch_embedding.weight"),
            "missing vision tower patch_embedding.weight"
        );
        assert!(
            name_set.contains("vision_tower.vision_model.embeddings.position_embedding.weight"),
            "missing vision tower position_embedding.weight"
        );
        assert!(
            name_set.contains("vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"),
            "missing vision tower encoder layer 0 q_proj"
        );
        assert!(
            name_set.contains("vision_tower.vision_model.post_layernorm.weight"),
            "missing vision tower post_layernorm"
        );

        // --- Projector (prefix: "multi_modal_projector") ---
        assert!(
            name_set.contains("multi_modal_projector.mm_input_projection_weight"),
            "missing projector mm_input_projection_weight"
        );
        assert!(
            name_set.contains("multi_modal_projector.mm_soft_emb_norm.weight"),
            "missing projector mm_soft_emb_norm.weight"
        );

        // --- Language model (prefix: "language_model.model") ---
        assert!(
            name_set.contains("language_model.model.embed_tokens.weight"),
            "missing language_model embed_tokens"
        );
        assert!(
            name_set.contains("language_model.model.layers.0.self_attn.q_proj.weight"),
            "missing language_model layer 0 q_proj"
        );
        assert!(
            name_set.contains("language_model.model.norm.weight"),
            "missing language_model final norm"
        );

        // --- Verify old/wrong prefixes do NOT exist ---
        assert!(
            !name_set.contains("model.vision_tower.vision_model.embeddings.patch_embedding.weight"),
            "old 'model.' prefix should not exist for vision tower"
        );
        assert!(
            !name_set.contains("model.multi_modal_projector.mm_input_projection_weight"),
            "old 'model.' prefix should not exist for projector"
        );
        assert!(
            !name_set.contains("model.embed_tokens.weight"),
            "bare 'model.' prefix should not exist (should be 'language_model.model.')"
        );
    }

    /// Same validation for the MLX community quantized model.
    #[test]
    fn test_weight_names_match_hf_checkpoint_mlx_gemma3_4b_4bit() {
        let model_dir = match resolve_hf_cache_dir("mlx-community/gemma-3-4b-it-4bit") {
            Some(d) => d,
            None => {
                eprintln!("SKIP: mlx-community/gemma-3-4b-it-4bit not cached locally");
                return;
            }
        };

        let names = collect_tensor_names(&model_dir);
        let name_set: std::collections::HashSet<&str> = names.iter().map(|s| s.as_str()).collect();

        // Vision tower (float, same prefix).
        assert!(
            name_set.contains("vision_tower.vision_model.embeddings.patch_embedding.weight"),
            "missing vision tower patch_embedding.weight"
        );

        // Projector (float).
        assert!(
            name_set.contains("multi_modal_projector.mm_input_projection_weight"),
            "missing projector mm_input_projection_weight"
        );
        assert!(
            name_set.contains("multi_modal_projector.mm_soft_emb_norm.weight"),
            "missing projector mm_soft_emb_norm.weight"
        );

        // Language model (quantized — has weight/biases/scales).
        assert!(
            name_set.contains("language_model.model.embed_tokens.weight"),
            "missing language_model embed_tokens.weight (quantized)"
        );
        assert!(
            name_set.contains("language_model.model.embed_tokens.scales"),
            "missing language_model embed_tokens.scales (quantized)"
        );
        assert!(
            name_set.contains("language_model.model.layers.0.self_attn.q_proj.weight"),
            "missing language_model layer 0 q_proj.weight"
        );
    }

    #[test]
    fn test_projector_forward_shape() {
        let dtype = DType::F32;
        let device = Device::Cpu;
        // 64 patches per side, kernel=4 → 16x16=256 output tokens
        let proj = Gemma3MultiModalProjector::zeros(1152, 2560, 64, 4, dtype, &device).unwrap();

        // Input: [1, 4096, 1152] (batch=1, 64*64 patches, vision_hidden=1152)
        let input = Tensor::zeros((1, 4096, 1152), dtype, &device).unwrap();
        let output = proj.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 256, 2560]); // [1, pooled_tokens, text_hidden]
    }

    #[test]
    fn test_config_from_json() {
        let json: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
                "image_token_index": 255999,
                "mm_tokens_per_image": 256,
                "vision_config": {
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "image_size": 896,
                    "patch_size": 14,
                    "layer_norm_eps": 1e-6
                },
                "text_config": {
                    "hidden_size": 2560,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 4,
                    "num_hidden_layers": 26,
                    "intermediate_size": 10240,
                    "vocab_size": 262144,
                    "head_dim": 256,
                    "query_pre_attn_scalar": 256,
                    "sliding_window_pattern": 2
                }
            }"#,
        )
        .unwrap();

        let config = Gemma3VisionModelConfig::from_hf_config(&json).unwrap();
        assert_eq!(config.vision_config.hidden_size, 1152);
        assert_eq!(config.vision_config.image_size, 896);
        assert_eq!(config.text_config.hidden_size, 2560);
        assert_eq!(config.image_token_index, 255999);
        assert_eq!(config.mm_tokens_per_image, 256);
        assert_eq!(config.projection_dim, 2560);
        assert_eq!(config.patches_per_image, 64); // 896/14
        assert_eq!(config.pool_kernel_size, 4); // 64/16
    }
}
