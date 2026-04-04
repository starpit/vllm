// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! ModernBERT encoder model for MLX.
//!
//! ModernBERT is an encoder-only model with:
//! - Bidirectional attention (no causal mask, no KV cache)
//! - RoPE (rotary position embeddings) — alternating global/local layers
//! - Pre-norm with LayerNorm (layer 0 skips attn_norm since embedding has LN)
//! - GeGLU MLP (gated GELU)
//! - Output: hidden states `[seq_len, hidden_size]` (no lm_head)

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops::indexing::IndexOp;

use crate::models::llama::{assign_weight, load_safetensors_weights};
use crate::models::siglip::{load_layernorm_weights, load_linear_weights};

use super::MlxModel;
use crate::cache::MlxKvCache;
use vllm_model::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f32,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub norm_bias: bool,
    pub head_dim: usize,
    pub global_rope_theta: f32,
    pub local_rope_theta: Option<f32>,
    pub global_attn_every_n_layers: usize,
    pub local_attention: Option<usize>,
}

impl ModernBertConfig {
    pub fn from_hf_config(config: &HfModelConfig) -> Result<Self, String> {
        let hidden_size = config
            .hidden_size
            .ok_or_else(|| "missing hidden_size".to_string())?;
        let num_attention_heads = config
            .num_attention_heads
            .ok_or_else(|| "missing num_attention_heads".to_string())?;

        let norm_eps = config
            .extra
            .get("norm_eps")
            .and_then(|v| v.as_f64())
            .or_else(|| config.extra.get("layer_norm_eps").and_then(|v| v.as_f64()))
            .unwrap_or(1e-5) as f32;

        let attention_bias = config
            .extra
            .get("attention_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mlp_bias = config
            .extra
            .get("mlp_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let norm_bias = config
            .extra
            .get("norm_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let global_rope_theta = config
            .extra
            .get("global_rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(160000.0) as f32;

        let local_rope_theta = config
            .extra
            .get("local_rope_theta")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);

        let global_attn_every_n_layers = config
            .extra
            .get("global_attn_every_n_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;

        let local_attention = config
            .extra
            .get("local_attention")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let head_dim = config
            .head_dim()
            .unwrap_or(hidden_size / num_attention_heads);

        Ok(Self {
            vocab_size: config
                .vocab_size
                .ok_or_else(|| "missing vocab_size".to_string())?,
            hidden_size,
            num_hidden_layers: config
                .num_hidden_layers
                .ok_or_else(|| "missing num_hidden_layers".to_string())?,
            num_attention_heads,
            intermediate_size: config
                .intermediate_size
                .ok_or_else(|| "missing intermediate_size".to_string())?,
            max_position_embeddings: config.max_position_embeddings.unwrap_or(8192),
            norm_eps,
            attention_bias,
            mlp_bias,
            norm_bias,
            head_dim,
            global_rope_theta,
            local_rope_theta,
            global_attn_every_n_layers,
            local_attention,
        })
    }
}

// ---------------------------------------------------------------------------
// ModernBertEmbeddings
// ---------------------------------------------------------------------------

struct ModernBertEmbeddings {
    tok_embeddings: nn::Embedding,
    norm: nn::LayerNorm,
}

impl ModernBertEmbeddings {
    fn new(config: &ModernBertConfig) -> Result<Self, Exception> {
        Ok(Self {
            tok_embeddings: nn::Embedding::new(
                config.vocab_size as i32,
                config.hidden_size as i32,
            )?,
            norm: nn::LayerNormBuilder::new(config.hidden_size as i32)
                .eps(config.norm_eps)
                .build()?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.tok_embeddings.weight,
            weights,
            &format!("{prefix}.tok_embeddings.weight"),
        );
        load_layernorm_weights(&mut self.norm, weights, &format!("{prefix}.norm"));
    }

    fn forward(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        let embeds = self.tok_embeddings.forward(input_ids)?;
        self.norm.forward(&embeds)
    }
}

// ---------------------------------------------------------------------------
// ModernBertAttention
// ---------------------------------------------------------------------------

struct ModernBertAttention {
    wqkv: nn::Linear,
    wo: nn::Linear,
    rope: nn::Rope,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl ModernBertAttention {
    fn new(config: &ModernBertConfig, layer_id: usize) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let qkv_size = hidden * 3;

        // Per-layer RoPE theta: global vs local layers.
        let rope_theta = if !layer_id.is_multiple_of(config.global_attn_every_n_layers) {
            config.local_rope_theta.unwrap_or(config.global_rope_theta)
        } else {
            config.global_rope_theta
        };

        Ok(Self {
            wqkv: nn::LinearBuilder::new(hidden, qkv_size)
                .bias(config.attention_bias)
                .build()?,
            wo: nn::LinearBuilder::new(hidden, hidden)
                .bias(config.attention_bias)
                .build()?,
            rope: {
                let mut r = nn::Rope::new(config.head_dim as i32);
                r.base = rope_theta;
                r
            },
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim,
            scale: (config.head_dim as f32).powf(-0.5),
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        load_linear_weights(&mut self.wqkv, weights, &format!("{prefix}.Wqkv"));
        load_linear_weights(&mut self.wo, weights, &format!("{prefix}.Wo"));
    }

    fn forward(&mut self, hidden_states: &Array, _positions: &Array) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);
        let num_heads = self.num_heads as i32;
        let head_dim = self.head_dim as i32;

        // Fused QKV projection → split into Q, K, V.
        let qkv = self.wqkv.forward(hidden_states)?;
        let hidden = (self.num_heads * self.head_dim) as i32;
        let q = qkv.index((.., ..hidden));
        let k = qkv.index((.., hidden..2 * hidden));
        let v = qkv.index((.., 2 * hidden..));

        // Reshape: [seq, hidden] → [1, num_heads, seq, head_dim]
        let q = q
            .reshape(&[seq_len, num_heads, head_dim])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let k = k
            .reshape(&[seq_len, num_heads, head_dim])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let v = v
            .reshape(&[seq_len, num_heads, head_dim])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // RoPE on Q and K. Encoder always starts at position 0.
        let q = self.rope.forward((&q, 0))?;
        let k = self.rope.forward((&k, 0))?;

        // Bidirectional attention (no causal mask).
        let attn_output = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, None)?;

        // Reshape back: [1, num_heads, seq, head_dim] → [seq, hidden]
        let attn_output = attn_output
            .squeeze_axes(&[0])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq_len, num_heads * head_dim])?;

        self.wo.forward(&attn_output)
    }
}

// ---------------------------------------------------------------------------
// ModernBertMLP (GeGLU)
// ---------------------------------------------------------------------------

struct ModernBertMlp {
    wi: nn::Linear,
    wo: nn::Linear,
}

impl ModernBertMlp {
    fn new(config: &ModernBertConfig) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let inter = config.intermediate_size as i32;

        Ok(Self {
            wi: nn::LinearBuilder::new(hidden, inter * 2)
                .bias(config.mlp_bias)
                .build()?,
            wo: nn::LinearBuilder::new(inter, hidden)
                .bias(config.mlp_bias)
                .build()?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        load_linear_weights(&mut self.wi, weights, &format!("{prefix}.Wi"));
        load_linear_weights(&mut self.wo, weights, &format!("{prefix}.Wo"));
    }

    fn forward(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        let wi_out = self.wi.forward(hidden_states)?;
        let inter = wi_out.dim(-1) / 2;
        let input = wi_out.index((.., ..inter));
        let gate = wi_out.index((.., inter..));
        let activated = nn::gelu(&input)?;
        let gated = activated.multiply(&gate)?;
        self.wo.forward(&gated)
    }
}

// ---------------------------------------------------------------------------
// ModernBertLayer
// ---------------------------------------------------------------------------

struct ModernBertLayer {
    attn_norm: Option<nn::LayerNorm>, // None for layer 0 (identity)
    attn: ModernBertAttention,
    mlp_norm: nn::LayerNorm,
    mlp: ModernBertMlp,
}

impl ModernBertLayer {
    fn new(config: &ModernBertConfig, layer_id: usize) -> Result<Self, Exception> {
        let attn_norm = if layer_id == 0 {
            None // Identity — embedding already applied LayerNorm
        } else {
            Some(
                nn::LayerNormBuilder::new(config.hidden_size as i32)
                    .eps(config.norm_eps)
                    .build()?,
            )
        };

        Ok(Self {
            attn_norm,
            attn: ModernBertAttention::new(config, layer_id)?,
            mlp_norm: nn::LayerNormBuilder::new(config.hidden_size as i32)
                .eps(config.norm_eps)
                .build()?,
            mlp: ModernBertMlp::new(config)?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        if let Some(ref mut norm) = self.attn_norm {
            load_layernorm_weights(norm, weights, &format!("{prefix}.attn_norm"));
        }
        self.attn.load_weights(weights, &format!("{prefix}.attn"));
        load_layernorm_weights(&mut self.mlp_norm, weights, &format!("{prefix}.mlp_norm"));
        self.mlp.load_weights(weights, &format!("{prefix}.mlp"));
    }

    fn forward(&mut self, hidden_states: &Array, positions: &Array) -> Result<Array, Exception> {
        // Pre-norm attention.
        let normed = if let Some(ref mut norm) = self.attn_norm {
            norm.forward(hidden_states)?
        } else {
            hidden_states.clone()
        };
        let attn_out = self.attn.forward(&normed, positions)?;
        let hidden_states = hidden_states.add(&attn_out)?;

        // Pre-norm MLP.
        let normed = self.mlp_norm.forward(&hidden_states)?;
        let mlp_out = self.mlp.forward(&normed)?;
        hidden_states.add(&mlp_out)
    }
}

// ---------------------------------------------------------------------------
// ModernBertModel
// ---------------------------------------------------------------------------

pub struct MlxModernBertModel {
    embeddings: ModernBertEmbeddings,
    layers: Vec<ModernBertLayer>,
    final_norm: nn::LayerNorm,
    num_layers: usize,
}

impl MlxModernBertModel {
    pub fn new(config: &ModernBertConfig) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(ModernBertLayer::new(config, i)?);
        }

        Ok(Self {
            embeddings: ModernBertEmbeddings::new(config)?,
            layers,
            final_norm: nn::LayerNormBuilder::new(config.hidden_size as i32)
                .eps(config.norm_eps)
                .build()?,
            num_layers: config.num_hidden_layers,
        })
    }

    pub fn load_weights(&mut self, weights: &HashMap<String, Array>) {
        self.embeddings.load_weights(weights, "embeddings");
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("layers.{i}"));
        }
        load_layernorm_weights(&mut self.final_norm, weights, "final_norm");
    }

    pub fn load(
        model_dir: &Path,
        config: &ModernBertConfig,
        _dtype: mlx_rs::Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut model = Self::new(config)?;
        let raw_weights = load_safetensors_weights(model_dir)?;
        // Some checkpoints (e.g. answerdotai/ModernBERT-base) have "model." prefix,
        // others (e.g. lightonai/GTE-ModernColBERT-v1) do not. Strip if present.
        let weights: HashMap<String, Array> =
            if raw_weights.contains_key("model.embeddings.tok_embeddings.weight") {
                raw_weights
                    .into_iter()
                    .map(|(k, v)| {
                        let stripped = k.strip_prefix("model.").unwrap_or(&k).to_string();
                        (stripped, v)
                    })
                    .collect()
            } else {
                raw_weights
            };
        model.load_weights(&weights);
        // Eval all loaded weights to materialize them.
        mlx_rs::transforms::eval(weights.values())?;
        Ok(model)
    }
}

impl MlxModel for MlxModernBertModel {
    fn forward(
        &mut self,
        _input_ids: &Array,
        _positions: &Array,
        _kv_cache: &mut MlxKvCache,
        _rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        Err(Exception::custom(
            "ModernBERT is an encoder-only model — use hidden_states() instead of forward()",
        ))
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut hidden_states = self.embeddings.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            hidden_states = layer.forward(&hidden_states, positions)?;
        }
        self.final_norm.forward(&hidden_states)
    }
}

// ---------------------------------------------------------------------------
// ColBERTModernBertModel — ModernBERT + linear projection for ColBERT
// ---------------------------------------------------------------------------

pub struct MlxColBERTModernBertModel {
    inner: MlxModernBertModel,
    /// Linear projection: [hidden_size, colbert_dim] (no bias, no activation).
    projection: nn::Linear,
    num_layers: usize,
}

impl MlxColBERTModernBertModel {
    pub fn load(
        model_dir: &Path,
        config: &ModernBertConfig,
        _dtype: mlx_rs::Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Load backbone.
        let mut inner = MlxModernBertModel::new(config)?;
        let weights = load_safetensors_weights(model_dir)?;
        inner.load_weights(&weights);
        mlx_rs::transforms::eval(weights.values())?;

        // Load projection from 1_Dense/model.safetensors.
        let dense_path = model_dir.join("1_Dense").join("model.safetensors");
        let projection = if dense_path.exists() {
            let dense_weights = Array::load_safetensors(&dense_path)?;
            // Key is "linear.weight" with shape [colbert_dim, hidden_size].
            let proj_weight = dense_weights
                .get("linear.weight")
                .ok_or("1_Dense/model.safetensors missing 'linear.weight'")?;
            let colbert_dim = proj_weight.shape()[0];
            let mut linear = nn::LinearBuilder::new(config.hidden_size as i32, colbert_dim)
                .bias(false)
                .build()?;
            assign_weight(&mut linear.weight, &dense_weights, "linear.weight");
            mlx_rs::transforms::eval(dense_weights.values())?;
            linear
        } else {
            return Err(format!(
                "ColBERTModernBertModel requires 1_Dense/model.safetensors in {model_dir:?}"
            )
            .into());
        };

        let num_layers = config.num_hidden_layers;
        Ok(Self {
            inner,
            projection,
            num_layers,
        })
    }
}

impl MlxModel for MlxColBERTModernBertModel {
    fn forward(
        &mut self,
        _input_ids: &Array,
        _positions: &Array,
        _kv_cache: &mut MlxKvCache,
        _rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        Err(Exception::custom(
            "ColBERTModernBERT is an encoder-only model — use hidden_states() instead",
        ))
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let hidden = self.inner.hidden_states(input_ids, positions)?;
        // Project: [num_tokens, hidden_size] → [num_tokens, colbert_dim]
        self.projection.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

pub fn create_mlx_colbert_modernbert(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: mlx_rs::Dtype,
) -> Result<Box<dyn MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let mb_config = ModernBertConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let model = MlxColBERTModernBertModel::load(model_dir, &mb_config, dtype)?;
    Ok(Box::new(model))
}

pub fn create_mlx_modernbert(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: mlx_rs::Dtype,
) -> Result<Box<dyn MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let mb_config = ModernBertConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let model = MlxModernBertModel::load(model_dir, &mb_config, dtype)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ModernBertConfig {
        ModernBertConfig {
            vocab_size: 256,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 128,
            max_position_embeddings: 512,
            norm_eps: 1e-5,
            attention_bias: false,
            mlp_bias: false,
            norm_bias: true,
            head_dim: 16,
            global_rope_theta: 160000.0,
            local_rope_theta: Some(10000.0),
            global_attn_every_n_layers: 3,
            local_attention: Some(128),
        }
    }

    #[test]
    fn test_embeddings_shape() {
        let config = test_config();
        let mut emb = ModernBertEmbeddings::new(&config).unwrap();
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[4]);
        let out = emb.forward(&input_ids).unwrap();
        mlx_rs::transforms::eval([&out]).unwrap();
        assert_eq!(out.shape(), &[4, 64]);
    }

    #[test]
    fn test_mlp_shape() {
        let config = test_config();
        let mut mlp = ModernBertMlp::new(&config).unwrap();
        let input = Array::zeros::<f32>(&[4, 64]).unwrap();
        let out = mlp.forward(&input).unwrap();
        mlx_rs::transforms::eval([&out]).unwrap();
        assert_eq!(out.shape(), &[4, 64]);
    }

    #[test]
    fn test_attention_bidirectional() {
        let config = test_config();
        let mut attn = ModernBertAttention::new(&config, 0).unwrap();
        let input = Array::ones::<f32>(&[4, 64]).unwrap();
        let positions = Array::from_slice(&[0i32, 1, 2, 3], &[4]);
        let out = attn.forward(&input, &positions).unwrap();
        mlx_rs::transforms::eval([&out]).unwrap();
        assert_eq!(out.shape(), &[4, 64]);
    }

    #[test]
    fn test_layer_shape() {
        let config = test_config();
        // Layer 0: identity attn_norm
        let mut layer0 = ModernBertLayer::new(&config, 0).unwrap();
        assert!(layer0.attn_norm.is_none());
        let input = Array::ones::<f32>(&[4, 64]).unwrap();
        let positions = Array::from_slice(&[0i32, 1, 2, 3], &[4]);
        let out = layer0.forward(&input, &positions).unwrap();
        mlx_rs::transforms::eval([&out]).unwrap();
        assert_eq!(out.shape(), &[4, 64]);

        // Layer 1: has attn_norm
        let layer1 = ModernBertLayer::new(&config, 1).unwrap();
        assert!(layer1.attn_norm.is_some());
    }

    #[test]
    fn test_full_model_shape() {
        let config = test_config();
        let mut model = MlxModernBertModel::new(&config).unwrap();
        assert_eq!(model.num_layers(), 2);

        let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[4]);
        let positions = Array::from_slice(&[0i32, 1, 2, 3], &[4]);
        let out = model.hidden_states(&input_ids, &positions).unwrap();
        mlx_rs::transforms::eval([&out]).unwrap();
        assert_eq!(out.shape(), &[4, 64]);
    }

    #[test]
    fn test_forward_returns_error() {
        let config = test_config();
        let mut model = MlxModernBertModel::new(&config).unwrap();
        let input_ids = Array::from_slice(&[1i32], &[1]);
        let positions = Array::from_slice(&[0i32], &[1]);
        let mut kv_cache: MlxKvCache = vec![];
        assert!(
            model
                .forward(&input_ids, &positions, &mut kv_cache, None)
                .is_err()
        );
    }

    #[test]
    fn test_rope_theta_per_layer() {
        let config = test_config();
        // Layer 0: 0 % 3 == 0 → global (160000)
        let attn0 = ModernBertAttention::new(&config, 0).unwrap();
        assert_eq!(attn0.rope.base, 160000.0);
        // Layer 1: 1 % 3 != 0 → local (10000)
        let attn1 = ModernBertAttention::new(&config, 1).unwrap();
        assert_eq!(attn1.rope.base, 10000.0);
        // Layer 3: 3 % 3 == 0 → global
        let attn3 = ModernBertAttention::new(&config, 3).unwrap();
        assert_eq!(attn3.rope.base, 160000.0);
    }

    #[test]
    fn test_colbert_wrapper_output_shape() {
        // ColBERT wrapper: hidden_states should produce [N, colbert_dim] not [N, hidden_size].
        let config = test_config(); // hidden_size=64
        let colbert_dim = 16;
        let inner = MlxModernBertModel::new(&config).unwrap();

        // Build projection: Linear(64 → 16, no bias).
        let projection = nn::LinearBuilder::new(config.hidden_size as i32, colbert_dim)
            .bias(false)
            .build()
            .unwrap();

        let mut colbert = MlxColBERTModernBertModel {
            inner,
            projection,
            num_layers: config.num_hidden_layers,
        };

        let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[4]);
        let positions = Array::from_slice(&[0i32, 1, 2, 3], &[4]);
        let out = colbert.hidden_states(&input_ids, &positions).unwrap();
        mlx_rs::transforms::eval([&out]).unwrap();
        // Output should be [4, 16] (projected), not [4, 64] (backbone).
        assert_eq!(out.shape(), &[4, colbert_dim]);
    }

    #[test]
    fn test_colbert_forward_returns_error() {
        let config = test_config();
        let projection = nn::LinearBuilder::new(config.hidden_size as i32, 16)
            .bias(false)
            .build()
            .unwrap();
        let mut colbert = MlxColBERTModernBertModel {
            inner: MlxModernBertModel::new(&config).unwrap(),
            projection,
            num_layers: config.num_hidden_layers,
        };
        let input_ids = Array::from_slice(&[1i32], &[1]);
        let positions = Array::from_slice(&[0i32], &[1]);
        let mut kv_cache: MlxKvCache = vec![];
        assert!(
            colbert
                .forward(&input_ids, &positions, &mut kv_cache, None)
                .is_err()
        );
    }

    #[test]
    fn test_config_from_hf() {
        let mut hf = HfModelConfig::default();
        hf.hidden_size = Some(768);
        hf.num_attention_heads = Some(12);
        hf.num_hidden_layers = Some(22);
        hf.intermediate_size = Some(1152);
        hf.vocab_size = Some(50368);
        hf.max_position_embeddings = Some(8192);
        hf.extra
            .insert("norm_eps".to_string(), serde_json::Value::from(1e-5));
        hf.extra.insert(
            "global_rope_theta".to_string(),
            serde_json::Value::from(160000.0),
        );
        hf.extra.insert(
            "global_attn_every_n_layers".to_string(),
            serde_json::json!(3),
        );

        let config = ModernBertConfig::from_hf_config(&hf).unwrap();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.num_hidden_layers, 22);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.global_attn_every_n_layers, 3);
    }
}
