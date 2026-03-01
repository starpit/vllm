// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Gemma 3 multimodal (vision-language) model for Candle.
//!
//! Implements `Gemma3ForConditionalGeneration` which wraps:
//! - `SiglipVisionModel` — vision encoder (Step 4)
//! - `Gemma3MultiModalProjector` — linear→GELU→linear projector
//! - `Gemma3ForCausalLM` — existing text-only language model
//!
//! The VLM model processes images through the vision encoder, projects them
//! into the LLM's embedding space, merges with text embeddings at placeholder
//! positions, and runs the language model backbone.

use candle_core::{DType, Device, Module, Tensor};

use vllm_common::multimodal::MultimodalData;
use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::Linear;
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::gemma3::{Gemma3Config, Gemma3ForCausalLM};
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
    vision_hidden_size: usize,
    /// Projection dimension (usually same as text hidden size).
    projection_dim: usize,
    /// Token ID for image placeholders.
    image_token_index: u32,
    /// Number of image tokens per image in the token sequence.
    mm_tokens_per_image: usize,
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

        Ok(Self {
            vision_config,
            text_config,
            vision_hidden_size,
            projection_dim,
            image_token_index,
            mm_tokens_per_image,
        })
    }
}

// ---------------------------------------------------------------------------
// Gemma3MultiModalProjector
// ---------------------------------------------------------------------------

/// Projects vision encoder hidden states into the LLM's embedding space.
///
/// Architecture: Linear → GELU → Linear
struct Gemma3MultiModalProjector {
    linear_1: Linear,
    linear_2: Linear,
}

impl Gemma3MultiModalProjector {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        vision_hidden_size: usize,
        projection_dim: usize,
        dtype: DType,
    ) -> ModelResult<Self> {
        let linear_1 = Linear::load(weights, &format!("{prefix}.linear_1"), dtype)?;
        let linear_2 = Linear::load(weights, &format!("{prefix}.linear_2"), dtype)?;

        // Verify dimensions.
        if linear_1.in_features() != vision_hidden_size {
            return Err(ModelError::Other(format!(
                "projector linear_1 in_features {} != vision_hidden_size {}",
                linear_1.in_features(),
                vision_hidden_size,
            )));
        }
        if linear_2.out_features() != projection_dim {
            return Err(ModelError::Other(format!(
                "projector linear_2 out_features {} != projection_dim {}",
                linear_2.out_features(),
                projection_dim,
            )));
        }

        Ok(Self { linear_1, linear_2 })
    }

    fn forward(&self, x: &Tensor) -> ModelResult<Tensor> {
        let h = self.linear_1.forward(x).map_err(ModelError::Candle)?;
        let h = h.gelu().map_err(ModelError::Candle)?;
        self.linear_2.forward(&h).map_err(ModelError::Candle)
    }

    #[cfg(test)]
    fn zeros(
        vision_hidden_size: usize,
        projection_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        Ok(Self {
            linear_1: Linear::zeros(vision_hidden_size, projection_dim, dtype, device)?,
            linear_2: Linear::zeros(projection_dim, projection_dim, dtype, device)?,
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
        let vision_tower = SiglipVisionModel::load(
            weights,
            "model.vision_tower.vision_model",
            &config.vision_config,
            dtype,
        )?;

        let multi_modal_projector = Gemma3MultiModalProjector::load(
            weights,
            "model.multi_modal_projector",
            config.vision_hidden_size,
            config.projection_dim,
            dtype,
        )?;

        let language_model =
            Gemma3ForCausalLM::load(weights, &config.text_config, dtype, device, 0, 1)?;

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

        // Project: [N, num_patches, vision_hidden] → [N, num_patches, text_hidden]
        // Flatten to 2D for Linear, then back to 3D.
        let (n_images, n_patches, _) = vision_outputs.dims3().map_err(ModelError::Candle)?;
        let flat = vision_outputs
            .reshape((n_images * n_patches, ()))
            .map_err(ModelError::Candle)?;
        let projected = self.multi_modal_projector.forward(&flat)?;
        let projected = projected
            .reshape((n_images, n_patches, ()))
            .map_err(ModelError::Candle)?;

        // Scatter image embeddings into text embeddings at placeholder positions.
        let mut merged = text_embeds;
        for (img_idx, placeholder) in mm_data.image_placeholders.iter().enumerate() {
            if img_idx >= n_images {
                break;
            }
            let image_embeds = projected.get(img_idx).map_err(ModelError::Candle)?; // [num_patches, hidden]

            // Replace tokens at offset..offset+length with image embeddings.
            let num_tokens = merged.dim(0).map_err(ModelError::Candle)?;
            let end = (placeholder.offset + placeholder.length).min(num_tokens);
            let actual_len = end.saturating_sub(placeholder.offset);
            if actual_len == 0 {
                continue;
            }

            // Trim image embeds if needed.
            let image_embeds = if actual_len < placeholder.length {
                image_embeds
                    .narrow(0, 0, actual_len)
                    .map_err(ModelError::Candle)?
            } else {
                image_embeds
                    .narrow(0, 0, actual_len.min(n_patches))
                    .map_err(ModelError::Candle)?
            };

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

    #[test]
    fn test_projector_forward_shape() {
        let dtype = DType::F32;
        let device = Device::Cpu;
        let proj = Gemma3MultiModalProjector::zeros(1152, 2304, dtype, &device).unwrap();

        let input = Tensor::zeros((256, 1152), dtype, &device).unwrap();
        let output = proj.forward(&input).unwrap();
        assert_eq!(output.dims(), &[256, 2304]);
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
                    "image_size": 224,
                    "patch_size": 14,
                    "layer_norm_eps": 1e-6
                },
                "text_config": {
                    "hidden_size": 2304,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 4,
                    "num_hidden_layers": 26,
                    "intermediate_size": 9216,
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
        assert_eq!(config.vision_config.image_size, 224);
        assert_eq!(config.text_config.hidden_size, 2304);
        assert_eq!(config.image_token_index, 255999);
        assert_eq!(config.mm_tokens_per_image, 256);
        assert_eq!(config.projection_dim, 2304);
    }
}
