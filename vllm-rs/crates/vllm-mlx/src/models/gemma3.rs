// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Gemma 3 text-only model architecture for MLX.
//!
//! Key differences from Gemma 2:
//! - No `attn_logit_softcapping` (removed)
//! - No `final_logit_softcapping` (removed)
//! - Per-head `q_norm` + `k_norm` (GemmaRmsNorm with +1 offset)
//! - `sliding_window_pattern: N` instead of explicit `layer_types` array
//! - Per-layer RoPE theta: `rope_theta` (global) vs `rope_local_base_freq` (local/sliding)
//! - Everything else identical: 4 norms, GemmaRmsNorm, GELU tanh, embed*sqrt(H), tied embeddings

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use crate::cache::MlxKvCache;
use crate::models::gemma2::assign_gemma_norm_weight;
use crate::models::llama::{assign_weight, load_safetensors_weights};
use crate::models::quantized_llama::{MlxEmbedTokens, QuantConfig, make_quantized_linear};
use vllm_model::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// MlxGemma3Config
// ---------------------------------------------------------------------------

/// Parsed configuration for a Gemma 3 model.
#[derive(Debug, Clone)]
pub struct MlxGemma3Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// RoPE theta for local/sliding-window layers.
    pub rope_local_base_freq: f32,
    pub head_dim: usize,
    /// Scaling factor for query before attention.
    pub query_pre_attn_scalar: f32,
    /// Optional soft cap for attention logits (None for Gemma3).
    pub attn_logit_softcapping: Option<f32>,
    /// Optional soft cap for final logits (None for Gemma3).
    pub final_logit_softcapping: Option<f32>,
    /// Whether attention projections use bias.
    pub attention_bias: bool,
    /// Sliding window size for local layers. None if not set.
    pub sliding_window: Option<usize>,
    /// Per-layer: `true` = sliding/local, `false` = global/full.
    pub layer_is_sliding: Vec<bool>,
}

impl MlxGemma3Config {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> Result<Self, String> {
        let hidden_size = config
            .hidden_size
            .ok_or_else(|| "missing hidden_size".to_string())?;
        let num_attention_heads = config.num_attention_heads.unwrap_or(8);
        // Gemma 3 default head_dim is 256 (may differ from hidden_size / num_heads).
        let head_dim = config.head_dim.unwrap_or(256);

        let query_pre_attn_scalar = config
            .extra
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_f64())
            .unwrap_or(head_dim as f64) as f32;

        let attn_logit_softcapping = config
            .extra
            .get("attn_logit_softcapping")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);

        let final_logit_softcapping = config
            .extra
            .get("final_logit_softcapping")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);

        let attention_bias = config
            .extra
            .get("attention_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let sliding_window = config
            .extra
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let rope_theta = config.rope_theta.unwrap_or(1_000_000.0) as f32;

        let rope_local_base_freq = config
            .extra
            .get("rope_local_base_freq")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0) as f32;

        let num_hidden_layers = config
            .num_hidden_layers
            .ok_or_else(|| "missing num_hidden_layers".to_string())?;

        // Gemma3 uses `sliding_window_pattern` to determine local vs global layers.
        let layer_is_sliding = if let Some(pattern) = config
            .extra
            .get("sliding_window_pattern")
            .and_then(|v| v.as_u64())
        {
            (0..num_hidden_layers)
                .map(|i| (i + 1) % (pattern as usize) != 0)
                .collect()
        } else if let Some(layer_types) = config.extra.get("layer_types").and_then(|v| v.as_array())
        {
            layer_types
                .iter()
                .map(|v| v.as_str() == Some("sliding_attention"))
                .collect()
        } else if sliding_window.is_some() {
            // Gemma 3 default: sliding_window_pattern=6
            (0..num_hidden_layers).map(|i| (i + 1) % 6 != 0).collect()
        } else {
            Vec::new()
        };

        Ok(Self {
            hidden_size,
            num_attention_heads,
            num_kv_heads: config.num_key_value_heads.unwrap_or(4),
            num_hidden_layers,
            intermediate_size: config
                .intermediate_size
                .ok_or_else(|| "missing intermediate_size".to_string())?,
            vocab_size: config.vocab_size.unwrap_or(262144),
            max_position_embeddings: config.max_position_embeddings.unwrap_or(131072),
            rms_norm_eps: config.norm_eps() as f32,
            rope_theta,
            rope_local_base_freq,
            head_dim,
            query_pre_attn_scalar,
            attn_logit_softcapping,
            final_logit_softcapping,
            attention_bias,
            sliding_window,
            layer_is_sliding,
        })
    }

    /// Returns the RoPE theta for a given layer.
    fn rope_theta_for_layer(&self, layer_idx: usize) -> f32 {
        if layer_idx < self.layer_is_sliding.len() && self.layer_is_sliding[layer_idx] {
            self.rope_local_base_freq
        } else {
            self.rope_theta
        }
    }
}

// ---------------------------------------------------------------------------
// MlxGemma3MLP
// ---------------------------------------------------------------------------

/// Gemma3 MLP (GELU-gated feed-forward) using MLX.
struct MlxGemma3MLP {
    gate_proj: nn::Linear,
    up_proj: nn::Linear,
    down_proj: nn::Linear,
}

impl MlxGemma3MLP {
    fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            up_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            down_proj: nn::LinearBuilder::new(intermediate_size, hidden_size)
                .bias(false)
                .build()?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.gate_proj.weight,
            weights,
            &format!("{prefix}.gate_proj.weight"),
        );
        assign_weight(
            &mut self.up_proj.weight,
            weights,
            &format!("{prefix}.up_proj.weight"),
        );
        assign_weight(
            &mut self.down_proj.weight,
            weights,
            &format!("{prefix}.down_proj.weight"),
        );
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::gelu_approximate(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// MlxGemma3Attention
// ---------------------------------------------------------------------------

/// Gemma3 attention with per-head QK norms and per-layer RoPE theta.
struct MlxGemma3Attention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
    q_norm: nn::RmsNorm,
    k_norm: nn::RmsNorm,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    /// Per-layer sliding window.
    sliding_window: Option<usize>,
}

impl MlxGemma3Attention {
    fn new(
        config: &MlxGemma3Config,
        layer_idx: usize,
        layer_sliding_window: Option<usize>,
    ) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let q_size = (config.num_attention_heads * config.head_dim) as i32;
        let kv_size = (config.num_kv_heads * config.head_dim) as i32;

        Ok(Self {
            q_proj: nn::LinearBuilder::new(hidden, q_size)
                .bias(config.attention_bias)
                .build()?,
            k_proj: nn::LinearBuilder::new(hidden, kv_size)
                .bias(config.attention_bias)
                .build()?,
            v_proj: nn::LinearBuilder::new(hidden, kv_size)
                .bias(config.attention_bias)
                .build()?,
            o_proj: nn::LinearBuilder::new(q_size, hidden)
                .bias(config.attention_bias)
                .build()?,
            q_norm: nn::RmsNormBuilder::new(config.head_dim as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            k_norm: nn::RmsNormBuilder::new(config.head_dim as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            rope: {
                let mut r = nn::Rope::new(config.head_dim as i32);
                r.base = config.rope_theta_for_layer(layer_idx);
                r
            },
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5),
            sliding_window: layer_sliding_window,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.q_proj.weight,
            weights,
            &format!("{prefix}.q_proj.weight"),
        );
        assign_weight(
            &mut self.k_proj.weight,
            weights,
            &format!("{prefix}.k_proj.weight"),
        );
        assign_weight(
            &mut self.v_proj.weight,
            weights,
            &format!("{prefix}.v_proj.weight"),
        );
        assign_weight(
            &mut self.o_proj.weight,
            weights,
            &format!("{prefix}.o_proj.weight"),
        );
        // QK norms with GemmaRmsNorm +1 offset.
        assign_gemma_norm_weight(
            &mut self.q_norm,
            weights,
            &format!("{prefix}.q_norm.weight"),
        );
        assign_gemma_norm_weight(
            &mut self.k_norm,
            weights,
            &format!("{prefix}.k_norm.weight"),
        );
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape: [seq, hidden] -> [seq, heads, head_dim]
        let q = q.reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?;
        let k = k.reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?;

        // Per-head QK norms (GemmaRmsNorm normalizes last dim = head_dim).
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose to [1, heads, seq, head_dim] for RoPE + SDPA.
        let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut v = v
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // RoPE
        let offset = if positions.size() > 0 {
            positions.reshape(&[-1])?.min(None)?.item::<i32>()
        } else {
            0
        };
        let q = self.rope.forward((&q, offset))?;
        k = self.rope.forward((&k, offset))?;

        // KV cache update
        if let Some((ck, cv)) = cache.take() {
            k = concatenate_axis(&[ck, k], 2)?;
            v = concatenate_axis(&[cv, v], 2)?;
        }
        *cache = Some((k.clone(), v.clone()));

        // Sliding window: trim K/V to last `w` positions.
        if let Some(w) = self.sliding_window {
            let kv_len = k.dim(2) as usize;
            if kv_len > w {
                let start = (kv_len - w) as i32;
                let end = kv_len as i32;
                k = k.try_index((.., .., start..end, ..))?;
                v = v.try_index((.., .., start..end, ..))?;
            }
        }

        // Fused SDPA
        let mask = if seq_len > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;

        // [1, heads, seq, head_dim] -> [seq, hidden]
        let hidden = (self.num_heads * self.head_dim) as i32;
        let out = out
            .squeeze_axes(&[0])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq_len, hidden])?;

        self.o_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// MlxGemma3DecoderLayer
// ---------------------------------------------------------------------------

/// A single Gemma3 decoder layer with 4 norms.
struct MlxGemma3DecoderLayer {
    self_attn: MlxGemma3Attention,
    mlp: MlxGemma3MLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
    pre_feedforward_layernorm: nn::RmsNorm,
    post_feedforward_layernorm: nn::RmsNorm,
}

impl MlxGemma3DecoderLayer {
    fn new(
        config: &MlxGemma3Config,
        layer_idx: usize,
        layer_sliding_window: Option<usize>,
    ) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: MlxGemma3Attention::new(config, layer_idx, layer_sliding_window)?,
            mlp: MlxGemma3MLP::new(config.hidden_size as i32, config.intermediate_size as i32)?,
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            pre_feedforward_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            post_feedforward_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        self.self_attn
            .load_weights(weights, &format!("{prefix}.self_attn"));
        self.mlp.load_weights(weights, &format!("{prefix}.mlp"));
        assign_gemma_norm_weight(
            &mut self.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        );
        assign_gemma_norm_weight(
            &mut self.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        );
        assign_gemma_norm_weight(
            &mut self.pre_feedforward_layernorm,
            weights,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
        );
        assign_gemma_norm_weight(
            &mut self.post_feedforward_layernorm,
            weights,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
        );
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        // Pre-attention norm + attention + post-attention norm + residual.
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, positions, cache)?;
        let attn_output = self.post_attention_layernorm.forward(&attn_output)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        // Pre-feedforward norm + MLP + post-feedforward norm + residual.
        let normed = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        let mlp_output = self.post_feedforward_layernorm.forward(&mlp_output)?;
        hidden_states.add(&mlp_output)
    }
}

// ---------------------------------------------------------------------------
// MlxGemma3ForCausalLM (float)
// ---------------------------------------------------------------------------

/// Gemma3 for causal language modeling using MLX (float weights).
pub struct MlxGemma3ForCausalLM {
    pub(crate) embed_tokens: nn::Embedding,
    layers: Vec<MlxGemma3DecoderLayer>,
    norm: nn::RmsNorm,
    pub(crate) normalizer: f32,
    final_logit_softcapping: Option<f32>,
    #[allow(dead_code)]
    pub(crate) config: MlxGemma3Config,
}

impl MlxGemma3ForCausalLM {
    fn new(config: &MlxGemma3Config) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_sliding_window =
                if i < config.layer_is_sliding.len() && config.layer_is_sliding[i] {
                    config.sliding_window
                } else {
                    None
                };
            layers.push(MlxGemma3DecoderLayer::new(config, i, layer_sliding_window)?);
        }

        Ok(Self {
            embed_tokens: nn::Embedding::new(config.vocab_size as i32, config.hidden_size as i32)?,
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            normalizer: (config.hidden_size as f32).sqrt(),
            final_logit_softcapping: config.final_logit_softcapping,
            config: config.clone(),
        })
    }

    /// Public constructor for use by the VLM wrapper.
    pub(crate) fn new_public(
        config: &MlxGemma3Config,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self::new(config)?)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>) {
        self.load_weights_with_prefix(weights, "model");
    }

    /// Load weights with a configurable prefix (e.g., "model" or "model.language_model.model").
    pub(crate) fn load_weights_with_prefix(
        &mut self,
        weights: &HashMap<String, Array>,
        prefix: &str,
    ) {
        assign_weight(
            &mut self.embed_tokens.weight,
            weights,
            &format!("{prefix}.embed_tokens.weight"),
        );
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("{prefix}.layers.{i}"));
        }
        assign_gemma_norm_weight(&mut self.norm, weights, &format!("{prefix}.norm.weight"));
    }

    fn load(
        model_dir: &Path,
        config: &MlxGemma3Config,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut model = Self::new(config)?;
        let weights = load_safetensors_weights(model_dir)?;
        model.load_weights(&weights);
        mlx_rs::transforms::eval(weights.values())?;
        Ok(model)
    }

    /// Embed token IDs and scale by sqrt(hidden_size).
    pub(crate) fn embed(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        let hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states.multiply(Array::from_f32(self.normalizer))
    }

    /// Run the transformer backbone on pre-computed embeddings, returning hidden states.
    pub(crate) fn backbone(
        &mut self,
        mut hidden_states: Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> Result<Array, Exception> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i])?;
        }
        self.norm.forward(&hidden_states)
    }

    /// Run backbone + logit projection (tied embeddings + optional softcapping).
    fn backbone_to_logits(
        &mut self,
        hidden_states: Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> Result<Array, Exception> {
        let hidden_states = self.backbone(hidden_states, positions, kv_cache)?;
        let logits = self.embed_tokens.as_linear(&hidden_states)?;
        let logits = if let Some(cap) = self.final_logit_softcapping {
            Self::apply_softcap(&logits, cap)?
        } else {
            logits
        };
        logits.as_dtype(Dtype::Float32)
    }

    /// Apply logit softcapping: `cap * tanh(logits / cap)`.
    fn apply_softcap(logits: &Array, cap: f32) -> Result<Array, Exception> {
        let scaled = logits.divide(Array::from_f32(cap))?;
        let capped = mlx_rs::ops::tanh(&scaled)?;
        capped.multiply(Array::from_f32(cap))
    }
}

impl super::MlxModel for MlxGemma3ForCausalLM {
    fn inject_lora(
        &mut self,
        adapter: &crate::lora::MlxLoraAdapter,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::lora::merge_lora_into_weight;
        let targets = &adapter.config.target_modules;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Attention projections.
            let attn_prefix = format!("model.layers.{}.self_attn", i);
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                if targets.iter().any(|t| t == name) {
                    let key = format!("{}.{}", attn_prefix, name);
                    if let Some((a, b)) = adapter.weights.get(&key) {
                        let proj = match name {
                            "q_proj" => &mut layer.self_attn.q_proj,
                            "k_proj" => &mut layer.self_attn.k_proj,
                            "v_proj" => &mut layer.self_attn.v_proj,
                            "o_proj" => &mut layer.self_attn.o_proj,
                            _ => unreachable!(),
                        };
                        merge_lora_into_weight(&mut proj.weight, a, b, adapter.scaling)?;
                    }
                }
            }
            // MLP projections.
            let mlp_prefix = format!("model.layers.{}.mlp", i);
            for name in ["gate_proj", "up_proj", "down_proj"] {
                if targets.iter().any(|t| t == name) {
                    let key = format!("{}.{}", mlp_prefix, name);
                    if let Some((a, b)) = adapter.weights.get(&key) {
                        let proj = match name {
                            "gate_proj" => &mut layer.mlp.gate_proj,
                            "up_proj" => &mut layer.mlp.up_proj,
                            "down_proj" => &mut layer.mlp.down_proj,
                            _ => unreachable!(),
                        };
                        merge_lora_into_weight(&mut proj.weight, a, b, adapter.scaling)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn forward(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> mlx_rs::error::Result<Array> {
        let hidden_states = self.embed(input_ids)?;
        self.backbone_to_logits(hidden_states, positions, kv_cache)
    }

    fn forward_embeds(
        &mut self,
        inputs_embeds: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> mlx_rs::error::Result<Array> {
        self.backbone_to_logits(inputs_embeds.clone(), positions, kv_cache)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let hidden_states = self.embed(input_ids)?;
        self.backbone(hidden_states, positions, &mut kv_cache)
    }
}

// ---------------------------------------------------------------------------
// Quantized Gemma3
// ---------------------------------------------------------------------------

/// Quantized Gemma3 MLP using QuantizedLinear.
struct MlxQuantizedGemma3MLP {
    gate_proj: nn::QuantizedLinear,
    up_proj: nn::QuantizedLinear,
    down_proj: nn::QuantizedLinear,
}

impl MlxQuantizedGemma3MLP {
    fn from_weights(weights: &HashMap<String, Array>, prefix: &str, qc: &QuantConfig) -> Self {
        Self {
            gate_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.gate_proj"),
                qc.group_size,
                qc.bits,
            ),
            up_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.up_proj"),
                qc.group_size,
                qc.bits,
            ),
            down_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.down_proj"),
                qc.group_size,
                qc.bits,
            ),
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::gelu_approximate(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden)
    }
}

/// Quantized Gemma3 attention.
struct MlxQuantizedGemma3Attention {
    q_proj: nn::QuantizedLinear,
    k_proj: nn::QuantizedLinear,
    v_proj: nn::QuantizedLinear,
    o_proj: nn::QuantizedLinear,
    q_norm: nn::RmsNorm,
    k_norm: nn::RmsNorm,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
}

impl MlxQuantizedGemma3Attention {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxGemma3Config,
        qc: &QuantConfig,
        layer_idx: usize,
        layer_sliding_window: Option<usize>,
    ) -> Result<Self, Exception> {
        // QK norms stay float (GemmaRmsNorm with +1 offset).
        let mut q_norm = nn::RmsNormBuilder::new(config.head_dim as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        let mut k_norm = nn::RmsNormBuilder::new(config.head_dim as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        assign_gemma_norm_weight(&mut q_norm, weights, &format!("{prefix}.q_norm.weight"));
        assign_gemma_norm_weight(&mut k_norm, weights, &format!("{prefix}.k_norm.weight"));

        Ok(Self {
            q_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.q_proj"),
                qc.group_size,
                qc.bits,
            ),
            k_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.k_proj"),
                qc.group_size,
                qc.bits,
            ),
            v_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.v_proj"),
                qc.group_size,
                qc.bits,
            ),
            o_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.o_proj"),
                qc.group_size,
                qc.bits,
            ),
            q_norm,
            k_norm,
            rope: {
                let mut r = nn::Rope::new(config.head_dim as i32);
                r.base = config.rope_theta_for_layer(layer_idx);
                r
            },
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5),
            sliding_window: layer_sliding_window,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [seq, heads, head_dim].
        let q = q.reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?;
        let k = k.reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?;

        // Per-head QK norms.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose to [1, heads, seq, head_dim].
        let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut v = v
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        let offset = if positions.size() > 0 {
            positions.reshape(&[-1])?.min(None)?.item::<i32>()
        } else {
            0
        };
        let q = self.rope.forward((&q, offset))?;
        k = self.rope.forward((&k, offset))?;

        if let Some((ck, cv)) = cache.take() {
            k = concatenate_axis(&[ck, k], 2)?;
            v = concatenate_axis(&[cv, v], 2)?;
        }
        *cache = Some((k.clone(), v.clone()));

        if let Some(w) = self.sliding_window {
            let kv_len = k.dim(2) as usize;
            if kv_len > w {
                let start = (kv_len - w) as i32;
                let end = kv_len as i32;
                k = k.try_index((.., .., start..end, ..))?;
                v = v.try_index((.., .., start..end, ..))?;
            }
        }

        let mask = if seq_len > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;

        let hidden = (self.num_heads * self.head_dim) as i32;
        let out = out
            .squeeze_axes(&[0])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq_len, hidden])?;

        self.o_proj.forward(&out)
    }
}

/// A single quantized Gemma3 decoder layer with 4 norms.
struct MlxQuantizedGemma3DecoderLayer {
    self_attn: MlxQuantizedGemma3Attention,
    mlp: MlxQuantizedGemma3MLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
    pre_feedforward_layernorm: nn::RmsNorm,
    post_feedforward_layernorm: nn::RmsNorm,
}

impl MlxQuantizedGemma3DecoderLayer {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &MlxGemma3Config,
        qc: &QuantConfig,
        layer_idx: usize,
        layer_sliding_window: Option<usize>,
    ) -> Result<Self, Exception> {
        let mut input_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        let mut post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        let mut pre_feedforward_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        let mut post_feedforward_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;

        assign_gemma_norm_weight(
            &mut input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        );
        assign_gemma_norm_weight(
            &mut post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        );
        assign_gemma_norm_weight(
            &mut pre_feedforward_layernorm,
            weights,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
        );
        assign_gemma_norm_weight(
            &mut post_feedforward_layernorm,
            weights,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
        );

        Ok(Self {
            self_attn: MlxQuantizedGemma3Attention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                qc,
                layer_idx,
                layer_sliding_window,
            )?,
            mlp: MlxQuantizedGemma3MLP::from_weights(weights, &format!("{prefix}.mlp"), qc),
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, positions, cache)?;
        let attn_output = self.post_attention_layernorm.forward(&attn_output)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        let mlp_output = self.post_feedforward_layernorm.forward(&mlp_output)?;
        hidden_states.add(&mlp_output)
    }
}

/// Quantized Gemma3 for causal language modeling using MLX.
pub struct MlxQuantizedGemma3ForCausalLM {
    embed_tokens: MlxEmbedTokens,
    layers: Vec<MlxQuantizedGemma3DecoderLayer>,
    norm: nn::RmsNorm,
    normalizer: f32,
    final_logit_softcapping: Option<f32>,
    #[allow(dead_code)]
    config: MlxGemma3Config,
}

impl MlxQuantizedGemma3ForCausalLM {
    fn load(
        model_dir: &Path,
        config: &MlxGemma3Config,
        qc: &QuantConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weights = load_safetensors_weights(model_dir)?;

        let embed_tokens =
            MlxEmbedTokens::from_weights(&weights, "model.embed_tokens", qc.group_size, qc.bits);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_sliding_window =
                if i < config.layer_is_sliding.len() && config.layer_is_sliding[i] {
                    config.sliding_window
                } else {
                    None
                };
            layers.push(MlxQuantizedGemma3DecoderLayer::from_weights(
                &weights,
                &format!("model.layers.{i}"),
                config,
                qc,
                i,
                layer_sliding_window,
            )?);
        }

        let mut norm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        assign_gemma_norm_weight(&mut norm, &weights, "model.norm.weight");

        mlx_rs::transforms::eval(weights.values())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            normalizer: (config.hidden_size as f32).sqrt(),
            final_logit_softcapping: config.final_logit_softcapping,
            config: config.clone(),
        })
    }
}

impl super::MlxModel for MlxQuantizedGemma3ForCausalLM {
    fn forward(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> mlx_rs::error::Result<Array> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i])?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        let logits = self.embed_tokens.as_linear(&hidden_states)?;

        let logits = if let Some(cap) = self.final_logit_softcapping {
            MlxGemma3ForCausalLM::apply_softcap(&logits, cap)?
        } else {
            logits
        };

        logits.as_dtype(Dtype::Float32)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = hidden_states.multiply(Array::from_f32(self.normalizer))?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i])?;
        }
        self.norm.forward(&hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/// Factory function for creating a float MLX Gemma3 model.
pub fn create_mlx_gemma3(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let gemma3_config = MlxGemma3Config::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let model = MlxGemma3ForCausalLM::load(model_dir, &gemma3_config, dtype)?;
    Ok(Box::new(model))
}

/// Factory function for creating a quantized MLX Gemma3 model.
pub fn create_mlx_quantized_gemma3(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let gemma3_config = MlxGemma3Config::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    tracing::info!(
        "Loading quantized MLX Gemma3 (group_size={}, bits={})",
        qc.group_size,
        qc.bits
    );
    let model = MlxQuantizedGemma3ForCausalLM::load(model_dir, &gemma3_config, &qc, dtype)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MlxModel;

    fn test_config() -> MlxGemma3Config {
        MlxGemma3Config {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            rope_local_base_freq: 10000.0,
            head_dim: 8,
            query_pre_attn_scalar: 8.0,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            attention_bias: false,
            sliding_window: Some(512),
            layer_is_sliding: vec![true, false],
        }
    }

    #[test]
    fn test_gemma3_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Gemma3ForCausalLM"],
                "hidden_size": 2304,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "num_hidden_layers": 26,
                "intermediate_size": 9216,
                "vocab_size": 262144,
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "rope_local_base_freq": 10000.0,
                "head_dim": 256,
                "query_pre_attn_scalar": 256,
                "sliding_window": 512,
                "sliding_window_pattern": 2,
                "attention_bias": false
            }"#,
        )
        .unwrap();

        let config = MlxGemma3Config::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 2304);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 256);
        assert!((config.query_pre_attn_scalar - 256.0).abs() < 1.0);
        assert!(config.attn_logit_softcapping.is_none());
        assert!(config.final_logit_softcapping.is_none());
        assert!(!config.attention_bias);
        assert_eq!(config.sliding_window, Some(512));
        assert!((config.rope_theta - 1000000.0).abs() < 1.0);
        assert!((config.rope_local_base_freq - 10000.0).abs() < 1.0);
        // sliding_window_pattern=2: alternating sliding/global
        assert_eq!(config.layer_is_sliding.len(), 26);
        assert!(config.layer_is_sliding[0]); // (0+1)%2=1 -> sliding
        assert!(!config.layer_is_sliding[1]); // (1+1)%2=0 -> global
    }

    #[test]
    fn test_gemma3_mlp_forward() {
        let mut mlp = MlxGemma3MLP::new(32, 64).unwrap();
        let x = mlx_rs::ops::ones::<f32>(&[3, 32]).unwrap();
        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[3, 32]);
    }

    #[test]
    fn test_gemma3_attention_forward() {
        let config = test_config();
        let mut attn = MlxGemma3Attention::new(&config, 0, None).unwrap();

        let x = mlx_rs::ops::ones::<f32>(&[4, 32]).unwrap();
        let positions = Array::from_iter(0..4i32, &[4]);
        let mut cache = None;

        let out = attn.forward(&x, &positions, &mut cache).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[4, 32]);
        assert!(cache.is_some());
    }

    #[test]
    fn test_gemma3_model_forward() {
        let config = test_config();
        let mut model = MlxGemma3ForCausalLM::new(&config).unwrap();

        let input_ids = Array::from_iter(0..3i32, &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache: MlxKvCache = vec![None; config.num_hidden_layers];

        let logits = model
            .forward(&input_ids, &positions, &mut kv_cache)
            .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);
    }

    #[test]
    fn test_gemma3_registry() {
        let registry = super::super::MlxModelRegistry::default_registry();
        assert!(registry.contains("Gemma3ForCausalLM"));
        assert!(registry.contains_quantized("Gemma3ForCausalLM"));
    }
}
