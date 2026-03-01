// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Gemma 3 multimodal (vision-language) model for MLX.
//!
//! Wraps:
//! - `MlxSiglipVisionModel` — vision encoder
//! - `MlxGemma3MultiModalProjector` — linear→GELU→linear projector
//! - `MlxGemma3ForCausalLM` — existing text-only language model
//!
//! All operations are lazy — vision encoder, projector, and LLM backbone
//! all build a single compute graph materialized by one `eval()` call.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use vllm_common::multimodal::MultimodalData;
use vllm_model::weight::HfModelConfig;

use crate::cache::MlxKvCache;
use crate::models::gemma3::{MlxGemma3Config, MlxGemma3ForCausalLM};
use crate::models::llama::{assign_weight, load_safetensors_weights};
use crate::models::siglip::{MlxSiglipVisionConfig, MlxSiglipVisionModel};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct MlxGemma3VisionModelConfig {
    vision_config: MlxSiglipVisionConfig,
    text_config: MlxGemma3Config,
    vision_hidden_size: usize,
    projection_dim: usize,
    mm_tokens_per_image: usize,
}

impl MlxGemma3VisionModelConfig {
    fn from_hf_config(
        config: &HfModelConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let vision_json = config
            .extra
            .get("vision_config")
            .ok_or("missing vision_config")?;
        let vision_config = MlxSiglipVisionConfig::from_json(vision_json)?;

        let text_config = if let Some(text_json) = config.extra.get("text_config") {
            let text_hf: HfModelConfig = serde_json::from_value(text_json.clone())?;
            MlxGemma3Config::from_hf_config(&text_hf)?
        } else {
            MlxGemma3Config::from_hf_config(config)?
        };

        let vision_hidden_size = vision_config.hidden_size;
        let projection_dim = text_config.hidden_size;

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
            mm_tokens_per_image,
        })
    }
}

// ---------------------------------------------------------------------------
// Projector
// ---------------------------------------------------------------------------

struct MlxGemma3MultiModalProjector {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl MlxGemma3MultiModalProjector {
    fn new(vision_hidden_size: usize, projection_dim: usize) -> Result<Self, Exception> {
        Ok(Self {
            linear_1: nn::LinearBuilder::new(vision_hidden_size as i32, projection_dim as i32)
                .bias(true)
                .build()?,
            linear_2: nn::LinearBuilder::new(projection_dim as i32, projection_dim as i32)
                .bias(true)
                .build()?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.linear_1.weight,
            weights,
            &format!("{prefix}.linear_1.weight"),
        );
        if let Some(b) = weights.get(&format!("{prefix}.linear_1.bias")) {
            self.linear_1.bias.value = Some(b.clone());
        }
        assign_weight(
            &mut self.linear_2.weight,
            weights,
            &format!("{prefix}.linear_2.weight"),
        );
        if let Some(b) = weights.get(&format!("{prefix}.linear_2.bias")) {
            self.linear_2.bias.value = Some(b.clone());
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = self.linear_1.forward(x)?;
        let h = nn::gelu_approximate(&h)?;
        self.linear_2.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// MlxGemma3ForConditionalGeneration
// ---------------------------------------------------------------------------

/// Gemma 3 multimodal model for MLX.
pub struct MlxGemma3ForConditionalGeneration {
    vision_tower: MlxSiglipVisionModel,
    multi_modal_projector: MlxGemma3MultiModalProjector,
    language_model: MlxGemma3ForCausalLM,
    #[allow(dead_code)]
    mm_tokens_per_image: usize,
    stashed_mm_data: Option<MultimodalData>,
}

impl MlxGemma3ForConditionalGeneration {
    fn new(
        config: &MlxGemma3VisionModelConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let vision_tower = MlxSiglipVisionModel::new(&config.vision_config)?;
        let multi_modal_projector =
            MlxGemma3MultiModalProjector::new(config.vision_hidden_size, config.projection_dim)?;
        let language_model = MlxGemma3ForCausalLM::new_public(&config.text_config)?;
        Ok(Self {
            vision_tower,
            multi_modal_projector,
            language_model,
            mm_tokens_per_image: config.mm_tokens_per_image,
            stashed_mm_data: None,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>) {
        self.vision_tower
            .load_weights(weights, "model.vision_tower.vision_model");
        self.multi_modal_projector
            .load_weights(weights, "model.multi_modal_projector");
        // Language model weights use "model.language_model.model." prefix in Gemma3 MM,
        // but the text model expects "model." prefix. We need to strip "language_model.".
        // However, the weights may already be in the correct format. Load with both patterns.
        self.language_model
            .load_weights_with_prefix(weights, "model");
    }

    /// Merge vision embeddings with text embeddings.
    fn merge_vision_embeddings(
        &mut self,
        input_ids: &Array,
        mm_data: &MultimodalData,
    ) -> Result<Array, Exception> {
        let text_embeds = self.language_model.embed(input_ids)?;

        if mm_data.images.is_empty() {
            return Ok(text_embeds);
        }

        // Build pixel tensor [N_images, 3, H, W].
        let pixel_arrays: Vec<Array> = mm_data
            .images
            .iter()
            .map(|img| {
                let arr =
                    Array::from_slice(&img.pixels, &[1, 3, img.height as i32, img.width as i32]);
                arr
            })
            .collect();
        let pixel_values = concatenate_axis(&pixel_arrays.iter().collect::<Vec<_>>(), 0)?;

        // Encode: [N, 3, H, W] → [N, num_patches, vision_hidden]
        let vision_outputs = self.vision_tower.forward(&pixel_values)?;

        // Project: [N, num_patches, vision_hidden] → [N, num_patches, text_hidden]
        let projected = self.multi_modal_projector.forward(&vision_outputs)?;

        // Scatter image embeddings into text embeddings at placeholder positions.
        let mut merged = text_embeds;
        for (img_idx, placeholder) in mm_data.image_placeholders.iter().enumerate() {
            if img_idx >= mm_data.images.len() {
                break;
            }
            let image_embeds = projected.try_index(img_idx as i32)?; // [num_patches, hidden]

            let num_tokens = merged.dim(0) as usize;
            let end = (placeholder.offset + placeholder.length).min(num_tokens);
            let actual_len = end.saturating_sub(placeholder.offset);
            if actual_len == 0 {
                continue;
            }

            let n_patches = image_embeds.dim(0) as usize;
            let use_len = actual_len.min(n_patches);
            let image_slice = image_embeds.try_index(..use_len as i32)?;

            // Build [before, image_embeds, after].
            let mut parts: Vec<Array> = Vec::new();
            if placeholder.offset > 0 {
                parts.push(merged.try_index(..placeholder.offset as i32)?);
            }
            parts.push(image_slice);
            if end < num_tokens {
                parts.push(merged.try_index(end as i32..)?);
            }
            merged = concatenate_axis(&parts.iter().collect::<Vec<_>>(), 0)?;
        }

        Ok(merged)
    }
}

impl super::MlxModel for MlxGemma3ForConditionalGeneration {
    fn forward(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> mlx_rs::error::Result<Array> {
        if let Some(mm_data) = self.stashed_mm_data.take() {
            let merged_embeds = self.merge_vision_embeddings(input_ids, &mm_data)?;
            self.language_model
                .forward_embeds(&merged_embeds, positions, kv_cache)
        } else {
            self.language_model.forward(input_ids, positions, kv_cache)
        }
    }

    fn forward_embeds(
        &mut self,
        inputs_embeds: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> mlx_rs::error::Result<Array> {
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

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

pub fn create_mlx_gemma3_mm(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let _ = dtype; // dtype resolved by model from config
    let mm_config = MlxGemma3VisionModelConfig::from_hf_config(config)?;
    let mut model = MlxGemma3ForConditionalGeneration::new(&mm_config)?;
    let weights = load_safetensors_weights(model_dir)?;
    model.load_weights(&weights);
    mlx_rs::transforms::eval(weights.values())?;
    Ok(Box::new(model))
}
