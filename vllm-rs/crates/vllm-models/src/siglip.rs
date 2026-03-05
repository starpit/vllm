// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! SigLIP vision encoder for Candle.
//!
//! Implements the vision tower used by Gemma 3 VLM (and other SigLIP-based models).
//! The encoder converts pixel values into a sequence of patch embeddings.
//!
//! Architecture:
//! - Patch embedding: unfold image into patches, linear project to embed_dim
//! - Learned position embedding
//! - Stack of encoder layers (LayerNorm → self-attention → LayerNorm → MLP)
//! - Optional post-encoder LayerNorm
//!
//! Key difference from decoder attention: vision encoder uses bidirectional
//! (full) attention — no causal mask.

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{LayerNorm, Linear};
use vllm_model::weight::ModelWeights;

/// Softmax along the last dimension.
fn softmax_last_dim(x: &Tensor) -> ModelResult<Tensor> {
    let max = x
        .max_keepdim(candle_core::D::Minus1)
        .map_err(ModelError::Candle)?;
    let shifted = x.broadcast_sub(&max).map_err(ModelError::Candle)?;
    let exp = shifted.exp().map_err(ModelError::Candle)?;
    let sum = exp
        .sum_keepdim(candle_core::D::Minus1)
        .map_err(ModelError::Candle)?;
    exp.broadcast_div(&sum).map_err(ModelError::Candle)
}

// ---------------------------------------------------------------------------
// SiglipVisionConfig
// ---------------------------------------------------------------------------

/// Parsed configuration for a SigLIP vision encoder.
#[derive(Debug, Clone)]
pub struct SiglipVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub layer_norm_eps: f64,
}

impl SiglipVisionConfig {
    /// Parse from the `vision_config` sub-object in a HuggingFace config.json.
    pub fn from_json(value: &serde_json::Value) -> ModelResult<Self> {
        let get_usize = |key: &str| -> ModelResult<usize> {
            value
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| ModelError::Other(format!("vision_config missing {key}")))
        };
        Ok(Self {
            hidden_size: get_usize("hidden_size")?,
            intermediate_size: get_usize("intermediate_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            num_attention_heads: get_usize("num_attention_heads")?,
            image_size: get_usize("image_size")?,
            patch_size: get_usize("patch_size")?,
            layer_norm_eps: value
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6),
        })
    }

    /// Number of patches: (image_size / patch_size)^2
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ---------------------------------------------------------------------------
// SiglipVisionEmbeddings
// ---------------------------------------------------------------------------

/// Patch embedding + learned position embedding.
struct SiglipVisionEmbeddings {
    /// Projects flattened patches `[3*P*P]` → `[hidden_size]`.
    patch_embedding: Linear,
    /// Learned position embedding `[num_patches, hidden_size]`.
    position_embedding: Tensor,
}

impl SiglipVisionEmbeddings {
    fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &SiglipVisionConfig,
        dtype: DType,
    ) -> ModelResult<Self> {
        let patch_embedding = Linear::load(weights, &format!("{prefix}.patch_embedding"), dtype)?;
        let position_embedding =
            weights.take_cast(&format!("{prefix}.position_embedding.weight"), dtype)?;

        // Verify shape.
        let expected_patches = config.num_patches();
        let shape = position_embedding.dims();
        if shape.len() != 2 || shape[0] != expected_patches || shape[1] != config.hidden_size {
            return Err(ModelError::Other(format!(
                "position_embedding shape mismatch: expected [{}, {}], got {:?}",
                expected_patches, config.hidden_size, shape
            )));
        }

        Ok(Self {
            patch_embedding,
            position_embedding,
        })
    }

    /// Forward: pixel_values `[B, 3, H, W]` → `[B, num_patches, hidden_size]`.
    fn forward(&self, pixel_values: &Tensor, config: &SiglipVisionConfig) -> ModelResult<Tensor> {
        let (batch, _channels, _h, _w) = pixel_values.dims4().map_err(ModelError::Candle)?;
        let p = config.patch_size;
        let grid = config.image_size / p;
        let num_patches = grid * grid;
        let patch_dim = 3 * p * p;

        // Unfold: [B, 3, H, W] → [B, num_patches, 3*P*P]
        // Reshape to [B, 3, grid, P, grid, P], permute to [B, grid, grid, 3, P, P], flatten patches.
        let x = pixel_values
            .reshape((batch, 3, grid, p, grid, p))
            .map_err(ModelError::Candle)?;
        // Permute: [B, 3, grid, P, grid, P] → [B, grid, grid, 3, P, P]
        let x = x.permute([0, 2, 4, 1, 3, 5]).map_err(ModelError::Candle)?;
        // Flatten to [B*num_patches, patch_dim] for Linear.
        let x = x
            .reshape((batch * num_patches, patch_dim))
            .map_err(ModelError::Candle)?;

        // Linear projection: [B*num_patches, patch_dim] → [B*num_patches, hidden_size].
        let embeddings = self
            .patch_embedding
            .forward(&x)
            .map_err(ModelError::Candle)?;

        // Reshape back to [B, num_patches, hidden_size].
        let embeddings = embeddings
            .reshape((batch, num_patches, config.hidden_size))
            .map_err(ModelError::Candle)?;

        // Add position embedding.
        embeddings
            .broadcast_add(&self.position_embedding)
            .map_err(ModelError::Candle)
    }

    fn zeros(config: &SiglipVisionConfig, dtype: DType, device: &Device) -> ModelResult<Self> {
        let patch_dim = 3 * config.patch_size * config.patch_size;
        let patch_embedding = Linear::zeros(patch_dim, config.hidden_size, dtype, device)?;
        let position_embedding =
            Tensor::zeros((config.num_patches(), config.hidden_size), dtype, device)
                .map_err(ModelError::Candle)?;
        Ok(Self {
            patch_embedding,
            position_embedding,
        })
    }
}

// ---------------------------------------------------------------------------
// SiglipEncoderLayer
// ---------------------------------------------------------------------------

/// A single SigLIP encoder layer: LayerNorm → self-attention → residual → LayerNorm → MLP → residual.
struct SiglipEncoderLayer {
    layer_norm1: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    layer_norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl SiglipEncoderLayer {
    fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &SiglipVisionConfig,
        dtype: DType,
    ) -> ModelResult<Self> {
        let ln1 = LayerNorm::load(
            weights,
            &format!("{prefix}.layer_norm1"),
            config.layer_norm_eps,
            dtype,
        )?;
        let q_proj = Linear::load(weights, &format!("{prefix}.self_attn.q_proj"), dtype)?;
        let k_proj = Linear::load(weights, &format!("{prefix}.self_attn.k_proj"), dtype)?;
        let v_proj = Linear::load(weights, &format!("{prefix}.self_attn.v_proj"), dtype)?;
        let out_proj = Linear::load(weights, &format!("{prefix}.self_attn.out_proj"), dtype)?;
        let ln2 = LayerNorm::load(
            weights,
            &format!("{prefix}.layer_norm2"),
            config.layer_norm_eps,
            dtype,
        )?;
        let fc1 = Linear::load(weights, &format!("{prefix}.mlp.fc1"), dtype)?;
        let fc2 = Linear::load(weights, &format!("{prefix}.mlp.fc2"), dtype)?;

        Ok(Self {
            layer_norm1: ln1,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            layer_norm2: ln2,
            fc1,
            fc2,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> ModelResult<Tensor> {
        let (batch, seq_len, hidden) = hidden_states.dims3().map_err(ModelError::Candle)?;
        let residual = hidden_states.clone();

        // Pre-attention norm (operates on last dim, works with 3D).
        let normed = self
            .layer_norm1
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;

        // Self-attention (bidirectional — no causal mask).
        let attn_output = self.self_attention(&normed, batch, seq_len)?;

        // Residual connection.
        let hidden_states = (residual + attn_output).map_err(ModelError::Candle)?;

        let residual = hidden_states.clone();

        // Pre-MLP norm.
        let normed = self
            .layer_norm2
            .forward(&hidden_states)
            .map_err(ModelError::Candle)?;

        // MLP: flatten [B, S, H] → [B*S, H] for Linear, then reshape back.
        let flat = normed
            .reshape((batch * seq_len, hidden))
            .map_err(ModelError::Candle)?;
        let mlp_output = self.fc1.forward(&flat).map_err(ModelError::Candle)?;
        let mlp_output = mlp_output.gelu().map_err(ModelError::Candle)?;
        let mlp_output = self.fc2.forward(&mlp_output).map_err(ModelError::Candle)?;
        let mlp_output = mlp_output
            .reshape((batch, seq_len, hidden))
            .map_err(ModelError::Candle)?;

        // Residual connection.
        (residual + mlp_output).map_err(ModelError::Candle)
    }

    /// Bidirectional multi-head self-attention.
    fn self_attention(&self, x: &Tensor, batch: usize, seq_len: usize) -> ModelResult<Tensor> {
        let hidden = self.num_heads * self.head_dim;

        // Flatten [B, S, H] → [B*S, H] for Linear projections.
        let flat = x
            .reshape((batch * seq_len, hidden))
            .map_err(ModelError::Candle)?;

        let q = self.q_proj.forward(&flat).map_err(ModelError::Candle)?;
        let k = self.k_proj.forward(&flat).map_err(ModelError::Candle)?;
        let v = self.v_proj.forward(&flat).map_err(ModelError::Candle)?;

        // Reshape to [B, num_heads, seq_len, head_dim].
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))
            .map_err(ModelError::Candle)?
            .transpose(1, 2)
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))
            .map_err(ModelError::Candle)?
            .transpose(1, 2)
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))
            .map_err(ModelError::Candle)?
            .transpose(1, 2)
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;

        // Scaled dot-product attention (no mask = bidirectional).
        let scale = (self.head_dim as f64).powf(-0.5);
        let k_tr = k
            .transpose(2, 3)
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;
        let scores = q.matmul(&k_tr).map_err(ModelError::Candle)?;
        let scores = (scores * scale).map_err(ModelError::Candle)?;

        // Upcast to f32 for softmax.
        let scores = scores.to_dtype(DType::F32).map_err(ModelError::Candle)?;
        let attn_weights = softmax_last_dim(&scores)?;
        let attn_weights = attn_weights
            .to_dtype(v.dtype())
            .map_err(ModelError::Candle)?;

        // Weighted sum of values.
        let attn_output = attn_weights.matmul(&v).map_err(ModelError::Candle)?;

        // Reshape back to [B*S, hidden_size] for out_proj.
        let attn_output = attn_output
            .transpose(1, 2)
            .map_err(ModelError::Candle)?
            .reshape((batch * seq_len, self.num_heads * self.head_dim))
            .map_err(ModelError::Candle)?;

        let out = self
            .out_proj
            .forward(&attn_output)
            .map_err(ModelError::Candle)?;

        // Reshape back to [B, S, H].
        out.reshape((batch, seq_len, hidden))
            .map_err(ModelError::Candle)
    }

    fn zeros(config: &SiglipVisionConfig, dtype: DType, device: &Device) -> ModelResult<Self> {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        Ok(Self {
            layer_norm1: LayerNorm::ones(h, config.layer_norm_eps, dtype, device)?,
            q_proj: Linear::zeros(h, h, dtype, device)?,
            k_proj: Linear::zeros(h, h, dtype, device)?,
            v_proj: Linear::zeros(h, h, dtype, device)?,
            out_proj: Linear::zeros(h, h, dtype, device)?,
            layer_norm2: LayerNorm::ones(h, config.layer_norm_eps, dtype, device)?,
            fc1: Linear::zeros(h, inter, dtype, device)?,
            fc2: Linear::zeros(inter, h, dtype, device)?,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
        })
    }
}

// ---------------------------------------------------------------------------
// SiglipVisionModel
// ---------------------------------------------------------------------------

/// SigLIP vision encoder: embeddings → encoder layers → post-layernorm.
///
/// Output: `[B, num_patches, hidden_size]` — a sequence of patch embeddings.
pub struct SiglipVisionModel {
    embeddings: SiglipVisionEmbeddings,
    encoder_layers: Vec<SiglipEncoderLayer>,
    post_layernorm: LayerNorm,
    config: SiglipVisionConfig,
}

impl SiglipVisionModel {
    /// Load from safetensors weights.
    pub fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &SiglipVisionConfig,
        dtype: DType,
    ) -> ModelResult<Self> {
        let embeddings =
            SiglipVisionEmbeddings::load(weights, &format!("{prefix}.embeddings"), config, dtype)?;

        let mut encoder_layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            encoder_layers.push(SiglipEncoderLayer::load(
                weights,
                &format!("{prefix}.encoder.layers.{i}"),
                config,
                dtype,
            )?);
        }

        let post_layernorm = LayerNorm::load(
            weights,
            &format!("{prefix}.post_layernorm"),
            config.layer_norm_eps,
            dtype,
        )?;

        Ok(Self {
            embeddings,
            encoder_layers,
            post_layernorm,
            config: config.clone(),
        })
    }

    /// Forward pass: pixel_values `[B, 3, H, W]` → `[B, num_patches, hidden_size]`.
    pub fn forward(&self, pixel_values: &Tensor) -> ModelResult<Tensor> {
        let mut hidden_states = self.embeddings.forward(pixel_values, &self.config)?;

        for layer in &self.encoder_layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        self.post_layernorm
            .forward(&hidden_states)
            .map_err(ModelError::Candle)
    }

    /// Vision hidden size.
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Create with zero weights (for testing).
    pub fn zeros(config: &SiglipVisionConfig, dtype: DType, device: &Device) -> ModelResult<Self> {
        let embeddings = SiglipVisionEmbeddings::zeros(config, dtype, device)?;
        let mut encoder_layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            encoder_layers.push(SiglipEncoderLayer::zeros(config, dtype, device)?);
        }
        let post_layernorm =
            LayerNorm::ones(config.hidden_size, config.layer_norm_eps, dtype, device)?;
        Ok(Self {
            embeddings,
            encoder_layers,
            post_layernorm,
            config: config.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vision_config() -> SiglipVisionConfig {
        SiglipVisionConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            image_size: 16,
            patch_size: 4,
            layer_norm_eps: 1e-6,
        }
    }

    #[test]
    fn test_config_num_patches() {
        let config = test_vision_config();
        assert_eq!(config.num_patches(), 16); // (16/4)^2 = 16
    }

    #[test]
    fn test_config_head_dim() {
        let config = test_vision_config();
        assert_eq!(config.head_dim(), 8); // 32/4 = 8
    }

    #[test]
    fn test_config_from_json() {
        let json: serde_json::Value = serde_json::from_str(
            r#"{
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "image_size": 224,
                "patch_size": 14,
                "layer_norm_eps": 1e-6
            }"#,
        )
        .unwrap();
        let config = SiglipVisionConfig::from_json(&json).unwrap();
        assert_eq!(config.hidden_size, 1152);
        assert_eq!(config.num_patches(), 256); // (224/14)^2
    }

    #[test]
    fn test_embeddings_forward_shape() {
        let config = test_vision_config();
        let device = Device::Cpu;
        let dtype = DType::F32;
        let emb = SiglipVisionEmbeddings::zeros(&config, dtype, &device).unwrap();

        let pixel_values =
            Tensor::zeros((1, 3, config.image_size, config.image_size), dtype, &device).unwrap();
        let output = emb.forward(&pixel_values, &config).unwrap();
        assert_eq!(
            output.dims(),
            &[1, config.num_patches(), config.hidden_size]
        );
    }

    #[test]
    fn test_encoder_layer_forward_shape() {
        let config = test_vision_config();
        let device = Device::Cpu;
        let dtype = DType::F32;
        let layer = SiglipEncoderLayer::zeros(&config, dtype, &device).unwrap();

        let input = Tensor::zeros(
            (1, config.num_patches(), config.hidden_size),
            dtype,
            &device,
        )
        .unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(
            output.dims(),
            &[1, config.num_patches(), config.hidden_size]
        );
    }

    #[test]
    fn test_vision_model_forward_shape() {
        let config = test_vision_config();
        let device = Device::Cpu;
        let dtype = DType::F32;
        let model = SiglipVisionModel::zeros(&config, dtype, &device).unwrap();

        let pixel_values =
            Tensor::zeros((1, 3, config.image_size, config.image_size), dtype, &device).unwrap();
        let output = model.forward(&pixel_values).unwrap();
        assert_eq!(
            output.dims(),
            &[1, config.num_patches(), config.hidden_size]
        );
    }

    #[test]
    fn test_vision_model_batch() {
        let config = test_vision_config();
        let device = Device::Cpu;
        let dtype = DType::F32;
        let model = SiglipVisionModel::zeros(&config, dtype, &device).unwrap();

        let pixel_values =
            Tensor::zeros((3, 3, config.image_size, config.image_size), dtype, &device).unwrap();
        let output = model.forward(&pixel_values).unwrap();
        assert_eq!(
            output.dims(),
            &[3, config.num_patches(), config.hidden_size]
        );
    }
}
