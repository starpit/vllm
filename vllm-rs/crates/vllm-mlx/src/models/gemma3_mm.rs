// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Gemma 3 multimodal (vision-language) model for MLX.
//!
//! Wraps:
//! - `MlxSiglipVisionModel` — vision encoder
//! - `MlxGemma3MultiModalProjector` — AvgPool2d → GemmaRMSNorm → matmul(projection_weight)
//! - `MlxGemma3ForCausalLM` — existing text-only language model
//!
//! All operations are lazy — vision encoder, projector, and LLM backbone
//! all build a single compute graph materialized by one `eval()` call.
//!
//! Weight prefix mapping (HF checkpoint → code):
//! - `vision_tower.vision_model.*` → vision encoder
//! - `multi_modal_projector.*` → projector
//! - `language_model.model.*` → text backbone

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
use crate::models::gemma2::assign_gemma_norm_weight;
use crate::models::gemma3::{MlxGemma3Config, MlxGemma3ForCausalLM, MlxQuantizedGemma3ForCausalLM};
use crate::models::llama::load_safetensors_weights;
use crate::models::quantized_llama::QuantConfig;
use crate::models::siglip::{MlxSiglipVisionConfig, MlxSiglipVisionModel};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct MlxGemma3VisionModelConfig {
    vision_config: MlxSiglipVisionConfig,
    text_config: MlxGemma3Config,
    #[allow(dead_code)]
    vision_hidden_size: usize,
    #[allow(dead_code)]
    projection_dim: usize,
    mm_tokens_per_image: usize,
    /// Number of patches per side: image_size / patch_size.
    patches_per_image: usize,
    /// AvgPool2d kernel size: patches_per_image / tokens_per_side.
    pool_kernel_size: usize,
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

        let patches_per_image = vision_config.image_size / vision_config.patch_size;
        let tokens_per_side = (mm_tokens_per_image as f64).sqrt() as usize;
        let pool_kernel_size = patches_per_image / tokens_per_side;

        Ok(Self {
            vision_config,
            text_config,
            vision_hidden_size,
            projection_dim,
            mm_tokens_per_image,
            patches_per_image,
            pool_kernel_size,
        })
    }
}

// ---------------------------------------------------------------------------
// Projector
// ---------------------------------------------------------------------------

/// Projects vision encoder hidden states into the LLM's embedding space.
///
/// Architecture: AvgPool2d → GemmaRMSNorm → matmul(projection_weight)
struct MlxGemma3MultiModalProjector {
    /// Raw projection weight `[vision_hidden, text_hidden]`.
    mm_input_projection_weight: Array,
    /// GemmaRMSNorm applied after pooling.
    mm_soft_emb_norm: nn::RmsNorm,
    /// Number of vision patches per side (image_size / patch_size).
    patches_per_image: usize,
    /// AvgPool2d kernel/stride size.
    kernel_size: usize,
}

impl MlxGemma3MultiModalProjector {
    fn new(
        vision_hidden_size: usize,
        patches_per_image: usize,
        kernel_size: usize,
        norm_eps: f32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            mm_input_projection_weight: Array::zeros::<f32>(&[
                vision_hidden_size as i32,
                1, // placeholder, replaced by load_weights
            ])?,
            mm_soft_emb_norm: nn::RmsNormBuilder::new(vision_hidden_size as i32)
                .eps(norm_eps)
                .build()?,
            patches_per_image,
            kernel_size,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        if let Some(w) = weights.get(&format!("{prefix}.mm_input_projection_weight")) {
            self.mm_input_projection_weight = w.clone();
        } else {
            tracing::warn!("Weight not found: {prefix}.mm_input_projection_weight");
        }
        // GemmaRMSNorm: weight += 1.0
        assign_gemma_norm_weight(
            &mut self.mm_soft_emb_norm,
            weights,
            &format!("{prefix}.mm_soft_emb_norm.weight"),
        );
    }

    /// Forward: vision_outputs `[B, num_patches, vision_hidden]` → `[B, pooled_tokens, text_hidden]`
    fn forward(&mut self, vision_outputs: &Array) -> Result<Array, Exception> {
        let shape = vision_outputs.shape();
        let batch = shape[0];
        let seq_length = shape[2]; // vision_hidden

        // Transpose: [B, num_patches, vision_hidden] → [B, vision_hidden, num_patches]
        let x = vision_outputs.transpose_axes(&[0, 2, 1])?;

        // Reshape to 2D grid: [B, vision_hidden, grid, grid]
        let grid = self.patches_per_image as i32;
        let x = x.reshape(&[batch, seq_length, grid, grid])?;

        // AvgPool2d: reshape blocks and mean.
        let k = self.kernel_size as i32;
        let out_grid = grid / k;
        // [B, C, grid, grid] → [B, C, out_grid, k, out_grid, k]
        let x = x.reshape(&[batch, seq_length, out_grid, k, out_grid, k])?;
        // Mean over kernel dims (axes 5 then 3).
        let x = x.mean_axis(5, None)?;
        let x = x.mean_axis(3, None)?;
        // Result: [B, C, out_grid, out_grid]

        // Flatten spatial: [B, C, out_grid, out_grid] → [B, C, pooled_tokens]
        let pooled_tokens = out_grid * out_grid;
        let x = x.reshape(&[batch, seq_length, pooled_tokens])?;

        // Transpose: [B, C, pooled_tokens] → [B, pooled_tokens, C]
        let x = x.transpose_axes(&[0, 2, 1])?;

        // GemmaRMSNorm (operates on last dim).
        let x = self.mm_soft_emb_norm.forward(&x)?;

        // Matmul with projection weight: [B, pooled_tokens, vision_hidden] @ [vision_hidden, text_hidden]
        x.matmul(&self.mm_input_projection_weight)
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
        let multi_modal_projector = MlxGemma3MultiModalProjector::new(
            config.vision_config.hidden_size,
            config.patches_per_image,
            config.pool_kernel_size,
            config.vision_config.layer_norm_eps,
        )?;
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
        // Vision tower — weights prefixed with "vision_tower.vision_model."
        self.vision_tower
            .load_weights(weights, "vision_tower.vision_model");
        // Projector — weights prefixed with "multi_modal_projector."
        self.multi_modal_projector
            .load_weights(weights, "multi_modal_projector");
        // Language model — weights prefixed with "language_model.model."
        self.language_model
            .load_weights_with_prefix(weights, "language_model.model");
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
            .map(|img| Array::from_slice(&img.pixels, &[1, 3, img.height as i32, img.width as i32]))
            .collect();
        let pixel_values = concatenate_axis(&pixel_arrays.iter().collect::<Vec<_>>(), 0)?;

        // Encode: [N, 3, H, W] → [N, num_patches, vision_hidden]
        let vision_outputs = self.vision_tower.forward(&pixel_values)?;

        // Project: [N, num_patches, vision_hidden] → [N, pooled_tokens, text_hidden]
        let projected = self.multi_modal_projector.forward(&vision_outputs)?;

        // Scatter image embeddings into text embeddings at placeholder positions.
        let mut merged = text_embeds;
        for (img_idx, placeholder) in mm_data.image_placeholders.iter().enumerate() {
            if img_idx >= mm_data.images.len() {
                break;
            }
            let image_embeds = projected.try_index(img_idx as i32)?; // [pooled_tokens, hidden]

            let num_tokens = merged.dim(0) as usize;
            let end = (placeholder.offset + placeholder.length).min(num_tokens);
            let actual_len = end.saturating_sub(placeholder.offset);
            if actual_len == 0 {
                continue;
            }

            let n_embed_tokens = image_embeds.dim(0) as usize;
            let use_len = actual_len.min(n_embed_tokens);
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
        _positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        if let Some(mm_data) = self.stashed_mm_data.take() {
            let merged_embeds = self.merge_vision_embeddings(input_ids, &mm_data)?;
            self.language_model
                .forward_embeds(&merged_embeds, _positions, kv_cache, rope_offset)
        } else {
            self.language_model
                .forward(input_ids, _positions, kv_cache, rope_offset)
        }
    }

    fn forward_embeds(
        &mut self,
        inputs_embeds: &Array,
        _positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        self.language_model
            .forward_embeds(inputs_embeds, _positions, kv_cache, rope_offset)
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

// ---------------------------------------------------------------------------
// Quantized VLM
// ---------------------------------------------------------------------------

/// Quantized Gemma 3 multimodal model for MLX.
///
/// Vision tower and projector remain float; language model uses quantized layers.
pub struct MlxQuantizedGemma3ForConditionalGeneration {
    vision_tower: MlxSiglipVisionModel,
    multi_modal_projector: MlxGemma3MultiModalProjector,
    language_model: MlxQuantizedGemma3ForCausalLM,
    #[allow(dead_code)]
    mm_tokens_per_image: usize,
    stashed_mm_data: Option<MultimodalData>,
}

impl MlxQuantizedGemma3ForConditionalGeneration {
    fn merge_vision_embeddings(
        &mut self,
        input_ids: &Array,
        mm_data: &MultimodalData,
    ) -> Result<Array, Exception> {
        let text_embeds = self.language_model.embed(input_ids)?;

        if mm_data.images.is_empty() {
            return Ok(text_embeds);
        }

        let pixel_arrays: Vec<Array> = mm_data
            .images
            .iter()
            .map(|img| Array::from_slice(&img.pixels, &[1, 3, img.height as i32, img.width as i32]))
            .collect();
        let pixel_values = concatenate_axis(&pixel_arrays.iter().collect::<Vec<_>>(), 0)?;

        let vision_outputs = self.vision_tower.forward(&pixel_values)?;
        let projected = self.multi_modal_projector.forward(&vision_outputs)?;

        let mut merged = text_embeds;
        for (img_idx, placeholder) in mm_data.image_placeholders.iter().enumerate() {
            if img_idx >= mm_data.images.len() {
                break;
            }
            let image_embeds = projected.try_index(img_idx as i32)?;

            let num_tokens = merged.dim(0) as usize;
            let end = (placeholder.offset + placeholder.length).min(num_tokens);
            let actual_len = end.saturating_sub(placeholder.offset);
            if actual_len == 0 {
                continue;
            }

            let n_embed_tokens = image_embeds.dim(0) as usize;
            let use_len = actual_len.min(n_embed_tokens);
            let image_slice = image_embeds.try_index(..use_len as i32)?;

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

impl super::MlxModel for MlxQuantizedGemma3ForConditionalGeneration {
    fn forward(
        &mut self,
        input_ids: &Array,
        _positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        let offset = rope_offset.unwrap_or(0);
        if let Some(mm_data) = self.stashed_mm_data.take() {
            let merged_embeds = self.merge_vision_embeddings(input_ids, &mm_data)?;
            self.language_model
                .forward_embeds(&merged_embeds, offset, kv_cache)
        } else {
            let hidden_states = self.language_model.embed(input_ids)?;
            self.language_model
                .forward_embeds(&hidden_states, offset, kv_cache)
        }
    }

    fn forward_embeds(
        &mut self,
        inputs_embeds: &Array,
        _positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        let offset = rope_offset.unwrap_or(0);
        self.language_model
            .forward_embeds(inputs_embeds, offset, kv_cache)
    }

    fn set_mm_data(&mut self, mm_data: Option<MultimodalData>) {
        self.stashed_mm_data = mm_data;
    }

    fn num_layers(&self) -> usize {
        self.language_model.num_layers()
    }
}

pub fn create_mlx_quantized_gemma3_mm(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let _ = dtype;
    let mm_config = MlxGemma3VisionModelConfig::from_hf_config(config)?;
    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    tracing::info!(
        "Loading quantized MLX Gemma3 VLM (group_size={}, bits={})",
        qc.group_size,
        qc.bits
    );

    let weights = load_safetensors_weights(model_dir)?;

    // Vision tower (float).
    let mut vision_tower = MlxSiglipVisionModel::new(&mm_config.vision_config)?;
    vision_tower.load_weights(&weights, "vision_tower.vision_model");

    // Projector (float).
    let mut multi_modal_projector = MlxGemma3MultiModalProjector::new(
        mm_config.vision_config.hidden_size,
        mm_config.patches_per_image,
        mm_config.pool_kernel_size,
        mm_config.vision_config.layer_norm_eps,
    )?;
    multi_modal_projector.load_weights(&weights, "multi_modal_projector");

    // Language model (quantized) with "language_model.model" prefix.
    let language_model = MlxQuantizedGemma3ForCausalLM::from_weights(
        &weights,
        "language_model.model",
        &mm_config.text_config,
        &qc,
    )?;

    mlx_rs::transforms::eval(weights.values())?;

    Ok(Box::new(MlxQuantizedGemma3ForConditionalGeneration {
        vision_tower,
        multi_modal_projector,
        language_model,
        mm_tokens_per_image: mm_config.mm_tokens_per_image,
        stashed_mm_data: None,
    }))
}
