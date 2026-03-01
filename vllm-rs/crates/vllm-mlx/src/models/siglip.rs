// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! SigLIP vision encoder for MLX.
//!
//! All operations are lazy — the vision encoder builds a compute graph that
//! fuses with the projector and LLM backbone for a single `eval()` call.

use std::collections::HashMap;

use mlx_rs::Array;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;

use crate::models::llama::assign_weight;

// ---------------------------------------------------------------------------
// MlxSiglipVisionConfig
// ---------------------------------------------------------------------------

/// Parsed configuration for a SigLIP vision encoder.
#[derive(Debug, Clone)]
pub struct MlxSiglipVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub layer_norm_eps: f32,
}

impl MlxSiglipVisionConfig {
    /// Parse from a `vision_config` JSON sub-object.
    pub fn from_json(
        value: &serde_json::Value,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let get = |key: &str| -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
            value
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| format!("vision_config missing {key}").into())
        };
        Ok(Self {
            hidden_size: get("hidden_size")?,
            intermediate_size: get("intermediate_size")?,
            num_hidden_layers: get("num_hidden_layers")?,
            num_attention_heads: get("num_attention_heads")?,
            image_size: get("image_size")?,
            patch_size: get("patch_size")?,
            layer_norm_eps: value
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6) as f32,
        })
    }

    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ---------------------------------------------------------------------------
// Helper: load LayerNorm weights
// ---------------------------------------------------------------------------

fn load_layernorm_weights(ln: &mut nn::LayerNorm, weights: &HashMap<String, Array>, prefix: &str) {
    if let Some(w) = weights.get(&format!("{prefix}.weight")) {
        ln.weight.value = Some(w.clone());
    }
    if let Some(b) = weights.get(&format!("{prefix}.bias")) {
        ln.bias.value = Some(b.clone());
    }
}

fn load_linear_weights(linear: &mut nn::Linear, weights: &HashMap<String, Array>, prefix: &str) {
    assign_weight(&mut linear.weight, weights, &format!("{prefix}.weight"));
    if let Some(b) = weights.get(&format!("{prefix}.bias")) {
        linear.bias.value = Some(b.clone());
    }
}

// ---------------------------------------------------------------------------
// MlxSiglipVisionEmbeddings
// ---------------------------------------------------------------------------

struct MlxSiglipVisionEmbeddings {
    patch_embedding: nn::Linear,
    position_embedding: Param<Array>,
}

impl MlxSiglipVisionEmbeddings {
    fn new(config: &MlxSiglipVisionConfig) -> Result<Self, Exception> {
        let patch_dim = 3 * config.patch_size * config.patch_size;
        let patch_embedding = nn::LinearBuilder::new(patch_dim as i32, config.hidden_size as i32)
            .bias(true)
            .build()?;
        let position_embedding = Param::new(Array::zeros::<f32>(&[
            config.num_patches() as i32,
            config.hidden_size as i32,
        ])?);
        Ok(Self {
            patch_embedding,
            position_embedding,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        load_linear_weights(
            &mut self.patch_embedding,
            weights,
            &format!("{prefix}.patch_embedding"),
        );
        if let Some(pe) = weights.get(&format!("{prefix}.position_embedding.weight")) {
            self.position_embedding.value = pe.clone();
        }
    }

    /// Forward: pixel_values `[B, 3, H, W]` → `[B, num_patches, hidden_size]`.
    fn forward(
        &mut self,
        pixel_values: &Array,
        config: &MlxSiglipVisionConfig,
    ) -> Result<Array, Exception> {
        let p = config.patch_size as i32;
        let grid = (config.image_size / config.patch_size) as i32;
        let num_patches = config.num_patches() as i32;
        let patch_dim = (3 * config.patch_size * config.patch_size) as i32;

        // pixel_values: [B, 3, H, W]
        let batch = pixel_values.dim(0);

        // Unfold: [B, 3, grid, P, grid, P]
        let x = pixel_values.reshape(&[batch, 3, grid, p, grid, p])?;
        // Permute to [B, grid, grid, 3, P, P]
        let x = x.transpose_axes(&[0, 2, 4, 1, 3, 5])?;
        // Flatten patches: [B, num_patches, 3*P*P]
        let x = x.reshape(&[batch, num_patches, patch_dim])?;

        // Project: [B, num_patches, hidden_size]
        // nn::Linear handles 3D input in MLX.
        let embeddings = self.patch_embedding.forward(&x)?;

        // Add position embedding.
        embeddings.add(&*self.position_embedding)
    }
}

// ---------------------------------------------------------------------------
// MlxSiglipEncoderLayer
// ---------------------------------------------------------------------------

struct MlxSiglipEncoderLayer {
    layer_norm1: nn::LayerNorm,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    layer_norm2: nn::LayerNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MlxSiglipEncoderLayer {
    fn new(config: &MlxSiglipVisionConfig) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        let inter = config.intermediate_size as i32;
        let eps = config.layer_norm_eps;

        Ok(Self {
            layer_norm1: nn::LayerNormBuilder::new(h).eps(eps).build()?,
            q_proj: nn::LinearBuilder::new(h, h).bias(true).build()?,
            k_proj: nn::LinearBuilder::new(h, h).bias(true).build()?,
            v_proj: nn::LinearBuilder::new(h, h).bias(true).build()?,
            out_proj: nn::LinearBuilder::new(h, h).bias(true).build()?,
            layer_norm2: nn::LayerNormBuilder::new(h).eps(eps).build()?,
            fc1: nn::LinearBuilder::new(h, inter).bias(true).build()?,
            fc2: nn::LinearBuilder::new(inter, h).bias(true).build()?,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        load_layernorm_weights(
            &mut self.layer_norm1,
            weights,
            &format!("{prefix}.layer_norm1"),
        );
        load_linear_weights(
            &mut self.q_proj,
            weights,
            &format!("{prefix}.self_attn.q_proj"),
        );
        load_linear_weights(
            &mut self.k_proj,
            weights,
            &format!("{prefix}.self_attn.k_proj"),
        );
        load_linear_weights(
            &mut self.v_proj,
            weights,
            &format!("{prefix}.self_attn.v_proj"),
        );
        load_linear_weights(
            &mut self.out_proj,
            weights,
            &format!("{prefix}.self_attn.out_proj"),
        );
        load_layernorm_weights(
            &mut self.layer_norm2,
            weights,
            &format!("{prefix}.layer_norm2"),
        );
        load_linear_weights(&mut self.fc1, weights, &format!("{prefix}.mlp.fc1"));
        load_linear_weights(&mut self.fc2, weights, &format!("{prefix}.mlp.fc2"));
    }

    fn forward(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        let residual = hidden_states.clone();

        // Pre-attention norm.
        let normed = self.layer_norm1.forward(hidden_states)?;

        // Self-attention (bidirectional).
        let attn_output = self.self_attention(&normed)?;

        // Residual.
        let hidden_states = residual.add(&attn_output)?;
        let residual = hidden_states.clone();

        // Pre-MLP norm.
        let normed = self.layer_norm2.forward(&hidden_states)?;

        // MLP: fc1 → GELU → fc2.
        let mlp_output = self.fc1.forward(&normed)?;
        let mlp_output = nn::gelu_approximate(&mlp_output)?;
        let mlp_output = self.fc2.forward(&mlp_output)?;

        // Residual.
        residual.add(&mlp_output)
    }

    /// Bidirectional multi-head self-attention.
    ///
    /// Input: `[B, S, H]`, output: `[B, S, H]`.
    fn self_attention(&mut self, x: &Array) -> Result<Array, Exception> {
        let batch = x.dim(0);
        let seq_len = x.dim(1);
        let num_heads = self.num_heads as i32;
        let head_dim = self.head_dim as i32;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [B, S, num_heads, head_dim] then transpose to [B, num_heads, S, head_dim].
        let q = q
            .reshape(&[batch, seq_len, num_heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[batch, seq_len, num_heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[batch, seq_len, num_heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Scaled dot-product attention (no mask = bidirectional).
        let scale = (self.head_dim as f32).powf(-0.5);
        let attn_output = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, None)?;

        // Reshape back to [B, S, hidden_size].
        let hidden = num_heads * head_dim;
        let attn_output = attn_output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, hidden])?;

        self.out_proj.forward(&attn_output)
    }
}

// ---------------------------------------------------------------------------
// MlxSiglipVisionModel
// ---------------------------------------------------------------------------

/// SigLIP vision encoder for MLX.
///
/// All operations are lazy — builds a compute graph that fuses with the
/// downstream projector and LLM backbone.
pub struct MlxSiglipVisionModel {
    embeddings: MlxSiglipVisionEmbeddings,
    encoder_layers: Vec<MlxSiglipEncoderLayer>,
    post_layernorm: nn::LayerNorm,
    config: MlxSiglipVisionConfig,
}

impl MlxSiglipVisionModel {
    /// Create with default initialization.
    pub fn new(config: &MlxSiglipVisionConfig) -> Result<Self, Exception> {
        let embeddings = MlxSiglipVisionEmbeddings::new(config)?;
        let mut encoder_layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            encoder_layers.push(MlxSiglipEncoderLayer::new(config)?);
        }
        let post_layernorm = nn::LayerNormBuilder::new(config.hidden_size as i32)
            .eps(config.layer_norm_eps)
            .build()?;
        Ok(Self {
            embeddings,
            encoder_layers,
            post_layernorm,
            config: config.clone(),
        })
    }

    /// Load weights from a flat HashMap.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        self.embeddings
            .load_weights(weights, &format!("{prefix}.embeddings"));
        for (i, layer) in self.encoder_layers.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("{prefix}.encoder.layers.{i}"));
        }
        load_layernorm_weights(
            &mut self.post_layernorm,
            weights,
            &format!("{prefix}.post_layernorm"),
        );
    }

    /// Forward pass: pixel_values `[B, 3, H, W]` → `[B, num_patches, hidden_size]`.
    pub fn forward(&mut self, pixel_values: &Array) -> Result<Array, Exception> {
        let mut hidden_states = self.embeddings.forward(pixel_values, &self.config)?;

        for layer in &mut self.encoder_layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        self.post_layernorm.forward(&hidden_states)
    }

    /// Vision hidden size.
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}
