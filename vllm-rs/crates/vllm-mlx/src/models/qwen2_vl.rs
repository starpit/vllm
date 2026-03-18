// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Qwen2-VL and Qwen2.5-VL multimodal (vision-language) model for MLX.
//!
//! Wraps:
//! - Vision encoder (custom ViT with 3D patch embed + 2D RoPE + PatchMerger)
//! - `MlxLlamaForCausalLM` / `MlxQuantizedLlamaForCausalLM` — Qwen2 text backbone
//!
//! Weight prefix mapping:
//! - `visual.*` → vision encoder
//! - `model.*` → text backbone
//! - `lm_head.*` → language model head

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
use crate::models::llama::{
    LlamaConfig, MlxLlamaForCausalLM, assign_weight, load_safetensors_weights,
};
use crate::models::quantized_llama::{MlxQuantizedLlamaForCausalLM, QuantConfig};
use crate::models::siglip::{load_layernorm_weights, load_linear_weights};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct MlxQwen2VisionConfig {
    hidden_size: usize,
    num_heads: usize,
    depth: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    in_channels: usize,
    spatial_merge_size: usize,
    is_qwen25: bool,
}

impl MlxQwen2VisionConfig {
    fn from_json(value: &serde_json::Value, is_qwen25: bool) -> Result<Self, String> {
        let get = |key: &str, default: usize| -> usize {
            value
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        Ok(Self {
            hidden_size: get("embed_dim", 1280),
            num_heads: get("num_heads", 16),
            depth: get("depth", 32),
            patch_size: get("patch_size", 14),
            temporal_patch_size: get("temporal_patch_size", 2),
            in_channels: get("in_channels", 3),
            spatial_merge_size: get("spatial_merge_size", 2),
            is_qwen25,
        })
    }

    #[allow(dead_code)]
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

// ---------------------------------------------------------------------------
// Vision blocks
// ---------------------------------------------------------------------------

/// Packed QKV attention for vision encoder.
struct MlxQwen2VisionAttention {
    qkv: nn::Linear,
    proj: nn::Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MlxQwen2VisionAttention {
    fn new(config: &MlxQwen2VisionConfig) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        Ok(Self {
            qkv: nn::LinearBuilder::new(h, 3 * h).bias(true).build()?,
            proj: nn::LinearBuilder::new(h, h).bias(true).build()?,
            num_heads: config.num_heads,
            head_dim: config.head_dim(),
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        load_linear_weights(&mut self.qkv, weights, &format!("{prefix}.qkv"));
        load_linear_weights(&mut self.proj, weights, &format!("{prefix}.proj"));
    }

    /// Bidirectional attention with optional 2D RoPE.
    fn forward(
        &mut self,
        x: &Array,
        rotary_cos: Option<&Array>,
        rotary_sin: Option<&Array>,
    ) -> Result<Array, Exception> {
        let seq_len = x.dim(0);
        let hidden = (self.num_heads * self.head_dim) as i32;
        let num_heads = self.num_heads as i32;
        let head_dim = self.head_dim as i32;

        let qkv = self.qkv.forward(x)?;
        let q = qkv.try_index((.., ..hidden))?;
        let k = qkv.try_index((.., hidden..2 * hidden))?;
        let v = qkv.try_index((.., 2 * hidden..))?;

        let q = q.reshape(&[seq_len, num_heads, head_dim])?;
        let k = k.reshape(&[seq_len, num_heads, head_dim])?;
        let v = v.reshape(&[seq_len, num_heads, head_dim])?;

        // Apply partial rotary embedding if provided.
        let (q, k) = if let (Some(cos), Some(sin)) = (rotary_cos, rotary_sin) {
            let rot_dim = cos.dim(-1);
            let q_rot = q.try_index((.., .., ..rot_dim))?;
            let q_pass = q.try_index((.., .., rot_dim..))?;
            let k_rot = k.try_index((.., .., ..rot_dim))?;
            let k_pass = k.try_index((.., .., rot_dim..))?;

            let q_rot = mlx_rope_apply(&q_rot, cos, sin)?;
            let k_rot = mlx_rope_apply(&k_rot, cos, sin)?;

            let q = concatenate_axis(&[&q_rot, &q_pass], -1)?;
            let k = concatenate_axis(&[&k_rot, &k_pass], -1)?;
            (q, k)
        } else {
            (q, k)
        };

        // Transpose to [1, num_heads, seq_len, head_dim] for scaled_dot_product_attention.
        let q = q
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[1, num_heads, seq_len, head_dim])?;
        let k = k
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[1, num_heads, seq_len, head_dim])?;
        let v = v
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[1, num_heads, seq_len, head_dim])?;

        let scale = (self.head_dim as f32).powf(-0.5);
        let attn_output =
            mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None::<&Array>)?;

        // Remove batch dim: [1, num_heads, seq_len, head_dim] → [seq_len, hidden]
        let attn_output = attn_output
            .reshape(&[num_heads, seq_len, head_dim])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq_len, hidden])?;

        self.proj.forward(&attn_output)
    }
}

/// Apply rotary embedding to a tensor (MLX).
/// x: [seq, heads, rot_dim], cos/sin: [seq, rot_dim]
fn mlx_rope_apply(x: &Array, cos: &Array, sin: &Array) -> Result<Array, Exception> {
    let half = x.dim(-1) / 2;
    let x1 = x.try_index((.., .., ..half))?;
    let x2 = x.try_index((.., .., half..))?;
    let neg_x2 = x2.negative()?;
    let x_rotated = concatenate_axis(&[&neg_x2, &x1], -1)?;

    // Unsqueeze cos/sin for broadcasting: [seq, rot_dim] → [seq, 1, rot_dim]
    let cos_b = cos.reshape(&[cos.dim(0), 1, cos.dim(-1)])?;
    let sin_b = sin.reshape(&[sin.dim(0), 1, sin.dim(-1)])?;

    x.multiply(&cos_b)?.add(&x_rotated.multiply(&sin_b)?)
}

/// Qwen2-VL vision block: LayerNorm + QuickGELU MLP.
struct MlxQwen2VisionBlock {
    norm1: nn::LayerNorm,
    attn: MlxQwen2VisionAttention,
    norm2: nn::LayerNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl MlxQwen2VisionBlock {
    fn new(config: &MlxQwen2VisionConfig) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        let intermediate = (config.hidden_size as f64 * 4.0) as i32;
        Ok(Self {
            norm1: nn::LayerNormBuilder::new(h).eps(1e-6).build()?,
            attn: MlxQwen2VisionAttention::new(config)?,
            norm2: nn::LayerNormBuilder::new(h).eps(1e-6).build()?,
            fc1: nn::LinearBuilder::new(h, intermediate).bias(true).build()?,
            fc2: nn::LinearBuilder::new(intermediate, h).bias(true).build()?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        load_layernorm_weights(&mut self.norm1, weights, &format!("{prefix}.norm1"));
        self.attn.load_weights(weights, &format!("{prefix}.attn"));
        load_layernorm_weights(&mut self.norm2, weights, &format!("{prefix}.norm2"));
        load_linear_weights(&mut self.fc1, weights, &format!("{prefix}.mlp.fc1"));
        load_linear_weights(&mut self.fc2, weights, &format!("{prefix}.mlp.fc2"));
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        rotary_cos: Option<&Array>,
        rotary_sin: Option<&Array>,
    ) -> Result<Array, Exception> {
        let residual = hidden_states.clone();
        let x = self.norm1.forward(hidden_states)?;
        let x = self.attn.forward(&x, rotary_cos, rotary_sin)?;
        let hidden_states = residual.add(&x)?;

        let residual = hidden_states.clone();
        let x = self.norm2.forward(&hidden_states)?;
        let x = self.fc1.forward(&x)?;
        // QuickGELU: x * sigmoid(1.702 * x)
        let scaled = x.multiply(Array::from_f32(1.702))?;
        let sigmoid = mlx_rs::ops::sigmoid(&scaled)?;
        let x = x.multiply(&sigmoid)?;
        let x = self.fc2.forward(&x)?;
        residual.add(&x)
    }
}

/// Qwen2.5-VL vision block: RMSNorm + SwiGLU MLP.
struct MlxQwen25VisionBlock {
    norm1: nn::RmsNorm,
    attn: MlxQwen2VisionAttention,
    norm2: nn::RmsNorm,
    gate_proj: nn::Linear,
    up_proj: nn::Linear,
    down_proj: nn::Linear,
}

impl MlxQwen25VisionBlock {
    fn new(config: &MlxQwen2VisionConfig) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        let intermediate = (config.hidden_size as f64 * 4.0) as i32;
        Ok(Self {
            norm1: nn::RmsNormBuilder::new(h).eps(1e-6).build()?,
            attn: MlxQwen2VisionAttention::new(config)?,
            norm2: nn::RmsNormBuilder::new(h).eps(1e-6).build()?,
            gate_proj: nn::LinearBuilder::new(h, intermediate)
                .bias(false)
                .build()?,
            up_proj: nn::LinearBuilder::new(h, intermediate)
                .bias(false)
                .build()?,
            down_proj: nn::LinearBuilder::new(intermediate, h)
                .bias(false)
                .build()?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        if let Some(w) = weights.get(&format!("{prefix}.norm1.weight")) {
            self.norm1.weight = mlx_rs::module::Param::new(w.clone());
        }
        self.attn.load_weights(weights, &format!("{prefix}.attn"));
        if let Some(w) = weights.get(&format!("{prefix}.norm2.weight")) {
            self.norm2.weight = mlx_rs::module::Param::new(w.clone());
        }
        load_linear_weights(
            &mut self.gate_proj,
            weights,
            &format!("{prefix}.mlp.gate_proj"),
        );
        load_linear_weights(&mut self.up_proj, weights, &format!("{prefix}.mlp.up_proj"));
        load_linear_weights(
            &mut self.down_proj,
            weights,
            &format!("{prefix}.mlp.down_proj"),
        );
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        rotary_cos: Option<&Array>,
        rotary_sin: Option<&Array>,
    ) -> Result<Array, Exception> {
        let residual = hidden_states.clone();
        let x = self.norm1.forward(hidden_states)?;
        let x = self.attn.forward(&x, rotary_cos, rotary_sin)?;
        let hidden_states = residual.add(&x)?;

        let residual = hidden_states.clone();
        let x = self.norm2.forward(&hidden_states)?;
        // SwiGLU: silu(gate_proj(x)) * up_proj(x) → down_proj
        let gate = self.gate_proj.forward(&x)?;
        let gate = nn::silu(&gate)?;
        let up = self.up_proj.forward(&x)?;
        let x = gate.multiply(&up)?;
        let x = self.down_proj.forward(&x)?;
        residual.add(&x)
    }
}

enum MlxVisionBlock {
    Qwen2(MlxQwen2VisionBlock),
    Qwen25(MlxQwen25VisionBlock),
}

impl MlxVisionBlock {
    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        match self {
            MlxVisionBlock::Qwen2(b) => b.load_weights(weights, prefix),
            MlxVisionBlock::Qwen25(b) => b.load_weights(weights, prefix),
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        rotary_cos: Option<&Array>,
        rotary_sin: Option<&Array>,
    ) -> Result<Array, Exception> {
        match self {
            MlxVisionBlock::Qwen2(b) => b.forward(hidden_states, rotary_cos, rotary_sin),
            MlxVisionBlock::Qwen25(b) => b.forward(hidden_states, rotary_cos, rotary_sin),
        }
    }
}

// ---------------------------------------------------------------------------
// PatchMerger
// ---------------------------------------------------------------------------

struct MlxQwen2VisionPatchMerger {
    ln_q: nn::LayerNorm,
    mlp_fc1: nn::Linear,
    mlp_fc2: nn::Linear,
    spatial_merge_size: usize,
}

impl MlxQwen2VisionPatchMerger {
    fn new(config: &MlxQwen2VisionConfig, text_hidden: usize) -> Result<Self, Exception> {
        let embed_dim = config.hidden_size as i32;
        let m = config.spatial_merge_size;
        let merged_dim = (m * m) as i32 * embed_dim;
        let text_hidden = text_hidden as i32;

        Ok(Self {
            ln_q: nn::LayerNormBuilder::new(embed_dim).eps(1e-6).build()?,
            mlp_fc1: nn::LinearBuilder::new(merged_dim, merged_dim)
                .bias(true)
                .build()?,
            mlp_fc2: nn::LinearBuilder::new(merged_dim, text_hidden)
                .bias(true)
                .build()?,
            spatial_merge_size: m,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        load_layernorm_weights(&mut self.ln_q, weights, &format!("{prefix}.ln_q"));
        load_linear_weights(&mut self.mlp_fc1, weights, &format!("{prefix}.mlp.0"));
        load_linear_weights(&mut self.mlp_fc2, weights, &format!("{prefix}.mlp.2"));
    }

    fn forward(&mut self, x: &Array, grid_h: usize, grid_w: usize) -> Result<Array, Exception> {
        let embed_dim = x.dim(-1) as usize;
        let m = self.spatial_merge_size;
        let out_h = grid_h / m;
        let out_w = grid_w / m;
        let merged_tokens = out_h * out_w;
        let merged_dim = m * m * embed_dim;

        // Reshape and permute for 2x2 merge.
        let x = x.reshape(&[
            out_h as i32,
            m as i32,
            out_w as i32,
            m as i32,
            embed_dim as i32,
        ])?;
        let x = x.transpose_axes(&[0, 2, 1, 3, 4])?;
        let x = x.reshape(&[merged_tokens as i32, merged_dim as i32])?;

        // LayerNorm on embed_dim.
        let x_for_norm = x.reshape(&[(merged_tokens * m * m) as i32, embed_dim as i32])?;
        let x_normed = self.ln_q.forward(&x_for_norm)?;
        let x = x_normed.reshape(&[merged_tokens as i32, merged_dim as i32])?;

        // MLP: Linear → GELU → Linear
        let x = self.mlp_fc1.forward(&x)?;
        let x = nn::gelu_approximate(&x)?;
        self.mlp_fc2.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// Vision Transformer
// ---------------------------------------------------------------------------

struct MlxQwen2VisionTransformer {
    patch_embed: nn::Linear,
    blocks: Vec<MlxVisionBlock>,
    merger: MlxQwen2VisionPatchMerger,
    head_dim: usize,
}

impl MlxQwen2VisionTransformer {
    fn new(config: &MlxQwen2VisionConfig, text_hidden: usize) -> Result<Self, Exception> {
        let patch_dim = (config.in_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size) as i32;
        let patch_embed = nn::LinearBuilder::new(patch_dim, config.hidden_size as i32)
            .bias(true)
            .build()?;

        let mut blocks = Vec::with_capacity(config.depth);
        for _ in 0..config.depth {
            let block = if config.is_qwen25 {
                MlxVisionBlock::Qwen25(MlxQwen25VisionBlock::new(config)?)
            } else {
                MlxVisionBlock::Qwen2(MlxQwen2VisionBlock::new(config)?)
            };
            blocks.push(block);
        }

        let merger = MlxQwen2VisionPatchMerger::new(config, text_hidden)?;

        Ok(Self {
            patch_embed,
            blocks,
            merger,
            head_dim: config.head_dim(),
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        // Patch embed weight may be 5D [C, pH, pW, T, embed_dim] from Conv3d —
        // reshape to 2D [C*pH*pW*T, embed_dim] for nn::Linear.
        let pe_key = format!("{prefix}.patch_embed.proj.weight");
        if let Some(w) = weights.get(&pe_key) {
            let shape = w.shape();
            if shape.len() > 2 {
                // Conv3d weight [embed_dim, T, pH, pW, C] → nn::Linear [embed_dim, T*pH*pW*C]
                let out_features = shape[0];
                let in_features: i32 = shape[1..].iter().product();
                if let Ok(reshaped) = w.reshape(&[out_features, in_features]) {
                    self.patch_embed.weight = mlx_rs::module::Param::new(reshaped);
                }
            } else {
                assign_weight(&mut self.patch_embed.weight, weights, &pe_key);
            }
        }
        if let Some(b) = weights.get(&format!("{prefix}.patch_embed.proj.bias")) {
            self.patch_embed.bias.value = Some(b.clone());
        }
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.load_weights(weights, &format!("{prefix}.blocks.{i}"));
        }
        self.merger
            .load_weights(weights, &format!("{prefix}.merger"));
    }

    fn forward(
        &mut self,
        pixel_values: &Array,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Array, Exception> {
        // Patch embedding.
        let mut hidden_states = self.patch_embed.forward(pixel_values)?;

        // Compute 2D RoPE.
        let seq_len = hidden_states.dim(0) as usize;
        let rot_dim = self.head_dim / 2; // partial_rotary_factor = 0.5

        // Build 2D position cos/sin (simplified — use sequential for now).
        let (cos, sin) = if seq_len == grid_h * grid_w && rot_dim > 0 {
            let half_rot = rot_dim / 2;
            let inv_freq: Vec<f32> = (0..half_rot)
                .map(|i| (1.0 / 10000.0f64.powf(2.0 * i as f64 / rot_dim as f64)) as f32)
                .collect();

            // Build y and x positions.
            let mut y_pos = Vec::with_capacity(seq_len);
            let mut x_pos = Vec::with_capacity(seq_len);
            for y in 0..grid_h {
                for x in 0..grid_w {
                    y_pos.push(y as f32);
                    x_pos.push(x as f32);
                }
            }

            let y_arr = Array::from_slice(&y_pos, &[seq_len as i32, 1]);
            let x_arr = Array::from_slice(&x_pos, &[seq_len as i32, 1]);
            let inv_arr = Array::from_slice(&inv_freq, &[1, half_rot as i32]);

            let freqs_y = y_arr.matmul(&inv_arr)?;
            let freqs_x = x_arr.matmul(&inv_arr)?;

            // Interleave: [y_freq, x_freq, y_freq, x_freq]
            let cos = concatenate_axis(
                &[
                    &freqs_y.cos()?,
                    &freqs_x.cos()?,
                    &freqs_y.cos()?,
                    &freqs_x.cos()?,
                ],
                -1,
            )?;
            let sin = concatenate_axis(
                &[
                    &freqs_y.sin()?,
                    &freqs_x.sin()?,
                    &freqs_y.sin()?,
                    &freqs_x.sin()?,
                ],
                -1,
            )?;
            (Some(cos), Some(sin))
        } else {
            (None, None)
        };

        // Run blocks.
        for block in &mut self.blocks {
            hidden_states = block.forward(&hidden_states, cos.as_ref(), sin.as_ref())?;
        }

        // Merge patches.
        self.merger.forward(&hidden_states, grid_h, grid_w)
    }
}

// ---------------------------------------------------------------------------
// Float VLM
// ---------------------------------------------------------------------------

pub struct MlxQwen2VLForConditionalGeneration {
    visual: MlxQwen2VisionTransformer,
    language_model: MlxLlamaForCausalLM,
    vision_config: MlxQwen2VisionConfig,
    stashed_mm_data: Option<MultimodalData>,
}

impl MlxQwen2VLForConditionalGeneration {
    fn merge_vision_embeddings(
        &mut self,
        input_ids: &Array,
        mm_data: &MultimodalData,
    ) -> Result<Array, Exception> {
        let text_embeds = self.language_model.embed(input_ids)?;

        if mm_data.images.is_empty() {
            return Ok(text_embeds);
        }

        let vc = &self.vision_config;
        let p = vc.patch_size;
        let tp = vc.temporal_patch_size;
        let c = vc.in_channels;

        let mut all_image_embeds = Vec::new();
        for img in &mm_data.images {
            let grid_h = img.height / p;
            let grid_w = img.width / p;
            let num_patches = grid_h * grid_w;

            // Build flattened patches from CHW pixels.
            let pixel_arr = Array::from_slice(
                &img.pixels,
                &[c as i32, img.height as i32, img.width as i32],
            );
            let x =
                pixel_arr.reshape(&[c as i32, grid_h as i32, p as i32, grid_w as i32, p as i32])?;
            let x = x.transpose_axes(&[1, 3, 0, 2, 4])?;
            let single_frame_dim = c * p * p;
            let x = x.reshape(&[num_patches as i32, single_frame_dim as i32])?;

            // Duplicate for temporal_patch_size.
            let x = if tp > 1 {
                let parts: Vec<&Array> = (0..tp).map(|_| &x).collect();
                concatenate_axis(&parts, -1)?
            } else {
                x
            };

            let image_embeds = self.visual.forward(&x, grid_h, grid_w)?;
            all_image_embeds.push(image_embeds);
        }

        // Scatter into text embeddings.
        let mut merged = text_embeds;
        for (img_idx, placeholder) in mm_data.image_placeholders.iter().enumerate() {
            if img_idx >= all_image_embeds.len() {
                break;
            }
            let image_embeds = &all_image_embeds[img_idx];
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

impl super::MlxModel for MlxQwen2VLForConditionalGeneration {
    fn forward(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        if let Some(mm_data) = self.stashed_mm_data.take() {
            let merged_embeds = self.merge_vision_embeddings(input_ids, &mm_data)?;
            self.language_model
                .forward_embeds(&merged_embeds, positions, kv_cache, rope_offset)
        } else {
            self.language_model
                .forward(input_ids, positions, kv_cache, rope_offset)
        }
    }

    fn forward_embeds(
        &mut self,
        inputs_embeds: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        self.language_model
            .forward_embeds(inputs_embeds, positions, kv_cache, rope_offset)
    }

    fn set_mm_data(&mut self, mm_data: Option<MultimodalData>) {
        self.stashed_mm_data = mm_data;
    }

    fn num_layers(&self) -> usize {
        self.language_model.num_layers()
    }
}

// ---------------------------------------------------------------------------
// Quantized VLM
// ---------------------------------------------------------------------------

pub struct MlxQuantizedQwen2VLForConditionalGeneration {
    visual: MlxQwen2VisionTransformer,
    language_model: MlxQuantizedLlamaForCausalLM,
    vision_config: MlxQwen2VisionConfig,
    stashed_mm_data: Option<MultimodalData>,
}

impl MlxQuantizedQwen2VLForConditionalGeneration {
    fn merge_vision_embeddings(
        &mut self,
        input_ids: &Array,
        mm_data: &MultimodalData,
    ) -> Result<Array, Exception> {
        let text_embeds = self.language_model.embed(input_ids)?;

        if mm_data.images.is_empty() {
            return Ok(text_embeds);
        }

        let vc = &self.vision_config;
        let p = vc.patch_size;
        let tp = vc.temporal_patch_size;
        let c = vc.in_channels;

        let mut all_image_embeds = Vec::new();
        for img in &mm_data.images {
            let grid_h = img.height / p;
            let grid_w = img.width / p;
            let num_patches = grid_h * grid_w;

            let pixel_arr = Array::from_slice(
                &img.pixels,
                &[c as i32, img.height as i32, img.width as i32],
            );
            let x =
                pixel_arr.reshape(&[c as i32, grid_h as i32, p as i32, grid_w as i32, p as i32])?;
            let x = x.transpose_axes(&[1, 3, 0, 2, 4])?;
            let single_frame_dim = c * p * p;
            let x = x.reshape(&[num_patches as i32, single_frame_dim as i32])?;

            let x = if tp > 1 {
                let parts: Vec<&Array> = (0..tp).map(|_| &x).collect();
                concatenate_axis(&parts, -1)?
            } else {
                x
            };

            let image_embeds = self.visual.forward(&x, grid_h, grid_w)?;
            all_image_embeds.push(image_embeds);
        }

        let mut merged = text_embeds;
        for (img_idx, placeholder) in mm_data.image_placeholders.iter().enumerate() {
            if img_idx >= all_image_embeds.len() {
                break;
            }
            let image_embeds = &all_image_embeds[img_idx];
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

impl super::MlxModel for MlxQuantizedQwen2VLForConditionalGeneration {
    fn forward(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        if let Some(mm_data) = self.stashed_mm_data.take() {
            let merged_embeds = self.merge_vision_embeddings(input_ids, &mm_data)?;
            self.language_model
                .forward_embeds(&merged_embeds, positions, kv_cache, rope_offset)
        } else {
            let hidden_states = self.language_model.embed(input_ids)?;
            self.language_model
                .forward_embeds(&hidden_states, positions, kv_cache, rope_offset)
        }
    }

    fn forward_embeds(
        &mut self,
        inputs_embeds: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        self.language_model
            .forward_embeds(inputs_embeds, positions, kv_cache, rope_offset)
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

/// Detect weight prefix convention for Qwen2-VL models.
///
/// MLX-community models use: `vision_tower.*`, `language_model.model.*`, `language_model.lm_head.*`
/// Original HF models use: `visual.*`, `model.*`, `lm_head.*`
fn detect_qwen2_vl_prefixes(
    weights: &HashMap<String, Array>,
) -> (&'static str, &'static str, &'static str) {
    if weights.keys().any(|k| k.starts_with("vision_tower.")) {
        (
            "vision_tower",
            "language_model.model",
            "language_model.lm_head",
        )
    } else {
        ("visual", "model", "lm_head")
    }
}

/// Dequantize vision encoder weights in-place.
///
/// MLX-community quantized Qwen2-VL models have quantized vision linear layers
/// (with `.scales` and `.biases`). We dequantize them at load time so the float
/// vision encoder can use them. Vision encoders are small enough that this is fine.
fn dequantize_vision_weights(
    weights: &mut HashMap<String, Array>,
    vis_prefix: &str,
    group_size: i32,
    bits: i32,
) {
    // Collect keys that need dequantization (have .scales).
    let quantized_bases: Vec<String> = weights
        .keys()
        .filter(|k| k.starts_with(vis_prefix) && k.ends_with(".scales"))
        .map(|k| k.strip_suffix(".scales").unwrap().to_string())
        .collect();

    for base in &quantized_bases {
        let w_key = format!("{base}.weight");
        let s_key = format!("{base}.scales");
        let b_key = format!("{base}.biases");

        if let (Some(w), Some(s)) = (weights.get(&w_key).cloned(), weights.get(&s_key).cloned()) {
            let biases = weights.get(&b_key).cloned();
            let dequant_result = if let Some(ref b) = biases {
                mlx_rs::ops::dequantize(&w, &s, b, group_size, bits)
            } else {
                let zero_biases = Array::zeros::<f32>(&[w.dim(0), 1]).unwrap_or(w.clone());
                mlx_rs::ops::dequantize(&w, &s, &zero_biases, group_size, bits)
            };
            if let Ok(dequant) = dequant_result {
                weights.insert(w_key, dequant);
                weights.remove(&s_key);
                weights.remove(&b_key);
            }
        }
    }
}

fn parse_config(
    config: &HfModelConfig,
    is_qwen25: bool,
) -> Result<(MlxQwen2VisionConfig, LlamaConfig), Box<dyn std::error::Error + Send + Sync>> {
    let vision_json = config
        .extra
        .get("vision_config")
        .ok_or("missing vision_config")?;
    let vision_config = MlxQwen2VisionConfig::from_json(vision_json, is_qwen25)?;

    let mut text_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    // Qwen2 defaults to rope_theta = 1M if not specified.
    if config.rope_theta.is_none() {
        text_config.rope_theta = 1_000_000.0;
    }

    Ok((vision_config, text_config))
}

pub fn create_mlx_qwen2_vl(
    model_dir: &Path,
    config: &HfModelConfig,
    _dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let (vision_config, text_config) = parse_config(config, false)?;

    let mut visual = MlxQwen2VisionTransformer::new(&vision_config, text_config.hidden_size)?;
    let mut language_model = MlxLlamaForCausalLM::new(&text_config)?;

    let weights = load_safetensors_weights(model_dir)?;
    // Detect weight prefix: MLX-community models use "vision_tower" + "language_model.model",
    // while original HF models use "visual" + "model".
    let (vis_prefix, lm_prefix, head_prefix) = detect_qwen2_vl_prefixes(&weights);
    visual.load_weights(&weights, vis_prefix);
    language_model.load_weights_with_prefix(&weights, lm_prefix);
    if let Some(ref mut lm_head) = language_model.lm_head {
        assign_weight(
            &mut lm_head.weight,
            &weights,
            &format!("{head_prefix}.weight"),
        );
    }

    mlx_rs::transforms::eval(weights.values())?;

    Ok(Box::new(MlxQwen2VLForConditionalGeneration {
        visual,
        language_model,
        vision_config,
        stashed_mm_data: None,
    }))
}

pub fn create_mlx_quantized_qwen2_vl(
    model_dir: &Path,
    config: &HfModelConfig,
    _dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let (vision_config, text_config) = parse_config(config, false)?;
    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();

    let mut weights = load_safetensors_weights(model_dir)?;
    let (vis_prefix, lm_prefix, _head_prefix) = detect_qwen2_vl_prefixes(&weights);

    // Dequantize vision encoder weights so the float vision encoder can use them.
    dequantize_vision_weights(&mut weights, vis_prefix, qc.group_size, qc.bits);

    let mut visual = MlxQwen2VisionTransformer::new(&vision_config, text_config.hidden_size)?;
    visual.load_weights(&weights, vis_prefix);

    let language_model = MlxQuantizedLlamaForCausalLM::from_weights_with_prefix(
        &weights,
        lm_prefix,
        &text_config,
        &qc,
    )?;

    mlx_rs::transforms::eval(weights.values())?;

    Ok(Box::new(MlxQuantizedQwen2VLForConditionalGeneration {
        visual,
        language_model,
        vision_config,
        stashed_mm_data: None,
    }))
}

pub fn create_mlx_qwen25_vl(
    model_dir: &Path,
    config: &HfModelConfig,
    _dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let (vision_config, text_config) = parse_config(config, true)?;

    let mut visual = MlxQwen2VisionTransformer::new(&vision_config, text_config.hidden_size)?;
    let mut language_model = MlxLlamaForCausalLM::new(&text_config)?;

    let weights = load_safetensors_weights(model_dir)?;
    let (vis_prefix, lm_prefix, head_prefix) = detect_qwen2_vl_prefixes(&weights);
    visual.load_weights(&weights, vis_prefix);
    language_model.load_weights_with_prefix(&weights, lm_prefix);
    if let Some(ref mut lm_head) = language_model.lm_head {
        assign_weight(
            &mut lm_head.weight,
            &weights,
            &format!("{head_prefix}.weight"),
        );
    }

    mlx_rs::transforms::eval(weights.values())?;

    Ok(Box::new(MlxQwen2VLForConditionalGeneration {
        visual,
        language_model,
        vision_config,
        stashed_mm_data: None,
    }))
}

pub fn create_mlx_quantized_qwen25_vl(
    model_dir: &Path,
    config: &HfModelConfig,
    _dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let (vision_config, text_config) = parse_config(config, true)?;
    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();

    let mut weights = load_safetensors_weights(model_dir)?;
    let (vis_prefix, lm_prefix, _head_prefix) = detect_qwen2_vl_prefixes(&weights);

    // Dequantize vision encoder weights so the float vision encoder can use them.
    dequantize_vision_weights(&mut weights, vis_prefix, qc.group_size, qc.bits);

    let mut visual = MlxQwen2VisionTransformer::new(&vision_config, text_config.hidden_size)?;
    visual.load_weights(&weights, vis_prefix);

    let language_model = MlxQuantizedLlamaForCausalLM::from_weights_with_prefix(
        &weights,
        lm_prefix,
        &text_config,
        &qc,
    )?;

    mlx_rs::transforms::eval(weights.values())?;

    Ok(Box::new(MlxQuantizedQwen2VLForConditionalGeneration {
        visual,
        language_model,
        vision_config,
        stashed_mm_data: None,
    }))
}
