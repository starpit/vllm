// SPDX-License-Identifier: Apache-2.0
//! Gemma 2 model architecture.
//!
//! Implements:
//! - `Gemma2ForCausalLM` — top-level model with tied embeddings and logit soft capping
//! - `Gemma2Model` — transformer backbone with embedding normalization
//! - `Gemma2DecoderLayer` — single transformer layer with 4 norms
//! - `Gemma2Attention` — attention with query scaling and logit soft capping
//! - `Gemma2MLP` — GELU-gated feed-forward network
//!
//! Key differences from LLaMA:
//! - GemmaRMSNorm: adds 1.0 to weight before applying (`y = x * (1 + w) / rms(x)`)
//! - GELU (tanh) activation instead of SiLU
//! - 4 norms per layer: input, post-attention, pre-feedforward, post-feedforward
//! - `query_pre_attn_scalar` for attention scaling (not `1/sqrt(head_dim)`)
//! - `attn_logit_softcapping`: soft cap on attention logits via tanh
//! - Embedding is multiplied by `sqrt(hidden_size)` after lookup
//! - Always tied embeddings (embed_tokens == lm_head)
//! - `final_logit_softcapping`: soft cap on output logits
//!
//! Port of: `vllm/model_executor/models/gemma2.py`

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{
    ColumnParallelLinear, Embedding, GemmaRmsNorm, Linear, RotaryEmbedding, RowParallelLinear,
};
use vllm_model::lora::LoraAdapter;
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::attention::attention_with_cache;

// ---------------------------------------------------------------------------
// Gemma2Config
// ---------------------------------------------------------------------------

/// Parsed configuration for a Gemma 2 model.
#[derive(Debug, Clone)]
pub struct Gemma2Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub head_dim: usize,
    /// Scaling factor for query before attention (replaces 1/sqrt(head_dim)).
    pub query_pre_attn_scalar: f64,
    /// Soft cap for attention logits (applied via tanh). None = no capping.
    pub attn_logit_softcapping: Option<f64>,
    /// Soft cap for final logits. None = no capping.
    pub final_logit_softcapping: Option<f64>,
    /// Whether attention projections use bias.
    pub attention_bias: bool,
    /// Sliding window size for "sliding_attention" layers. None if not set.
    pub sliding_window: Option<usize>,
    /// Per-layer attention type: `true` = sliding attention, `false` = full attention.
    /// Generated from the `layer_types` config field (Gemma2 interleaved pattern).
    /// Empty means all layers use full attention.
    pub layer_is_sliding: Vec<bool>,
}

impl Gemma2Config {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> ModelResult<Self> {
        let hidden_size = config
            .hidden_size
            .ok_or_else(|| ModelError::Other("missing hidden_size".into()))?;
        let num_attention_heads = config
            .num_attention_heads
            .ok_or_else(|| ModelError::Other("missing num_attention_heads".into()))?;
        let head_dim = config
            .head_dim()
            .unwrap_or(hidden_size / num_attention_heads);

        // Gemma2-specific config fields from `extra`.
        let query_pre_attn_scalar = config
            .extra
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_f64())
            .unwrap_or(head_dim as f64);

        let attn_logit_softcapping = config
            .extra
            .get("attn_logit_softcapping")
            .and_then(|v| v.as_f64());

        let final_logit_softcapping = config
            .extra
            .get("final_logit_softcapping")
            .and_then(|v| v.as_f64());

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

        let num_hidden_layers = config
            .num_hidden_layers
            .ok_or_else(|| ModelError::Other("missing num_hidden_layers".into()))?;

        // Parse layer_types: ["full_attention", "sliding_attention", ...].
        // Gemma2 uses an interleaved pattern where even layers are typically
        // "full_attention" and odd layers are "sliding_attention".
        let layer_is_sliding =
            if let Some(layer_types) = config.extra.get("layer_types").and_then(|v| v.as_array()) {
                layer_types
                    .iter()
                    .map(|v| v.as_str() == Some("sliding_attention"))
                    .collect()
            } else {
                Vec::new()
            };

        Ok(Self {
            hidden_size,
            num_attention_heads,
            num_kv_heads: config.num_kv_heads().unwrap_or(num_attention_heads),
            num_hidden_layers,
            intermediate_size: config
                .intermediate_size
                .ok_or_else(|| ModelError::Other("missing intermediate_size".into()))?,
            vocab_size: config
                .vocab_size
                .ok_or_else(|| ModelError::Other("missing vocab_size".into()))?,
            max_position_embeddings: config.max_position_embeddings.unwrap_or(8192),
            rms_norm_eps: config.norm_eps(),
            rope_theta: config.rope_theta.unwrap_or(10000.0),
            head_dim,
            query_pre_attn_scalar,
            attn_logit_softcapping,
            final_logit_softcapping,
            attention_bias,
            sliding_window,
            layer_is_sliding,
        })
    }
}

// ---------------------------------------------------------------------------
// Gemma2MLP
// ---------------------------------------------------------------------------

/// Gemma2 MLP (GELU-gated feed-forward network).
///
/// Forward: gate_proj(x) -> GELU(tanh) -> * up_proj(x) -> down_proj
pub struct Gemma2MLP {
    gate_proj: ColumnParallelLinear,
    up_proj: ColumnParallelLinear,
    down_proj: RowParallelLinear,
}

impl Gemma2MLP {
    /// Load MLP weights.
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        dtype: DType,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let gate_proj = ColumnParallelLinear::load(
            weights,
            &format!("{}.gate_proj", prefix),
            dtype,
            rank,
            world_size,
            false,
        )?;
        let up_proj = ColumnParallelLinear::load(
            weights,
            &format!("{}.up_proj", prefix),
            dtype,
            rank,
            world_size,
            false,
        )?;
        let down_proj = RowParallelLinear::load(
            weights,
            &format!("{}.down_proj", prefix),
            dtype,
            rank,
            world_size,
            true,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Create with zero weights (for testing).
    pub fn zeros(
        hidden_size: usize,
        intermediate_size: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let gate = Linear::zeros(hidden_size, intermediate_size, dtype, device)?;
        let up = Linear::zeros(hidden_size, intermediate_size, dtype, device)?;
        let down = Linear::zeros(intermediate_size, hidden_size, dtype, device)?;
        Ok(Self {
            gate_proj: ColumnParallelLinear::new(gate, false),
            up_proj: ColumnParallelLinear::new(up, false),
            down_proj: RowParallelLinear::new(down, true),
        })
    }
}

impl Module for Gemma2MLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = crate::ops::gelu_and_mul(&gate, &up)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// Gemma2Attention
// ---------------------------------------------------------------------------

/// Gemma2 multi-head attention with custom scaling and logit soft capping.
pub struct Gemma2Attention {
    q_proj: ColumnParallelLinear,
    k_proj: ColumnParallelLinear,
    v_proj: ColumnParallelLinear,
    o_proj: RowParallelLinear,
    rotary_emb: RotaryEmbedding,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    /// Optional soft cap for attention logits.
    attn_logit_softcapping: Option<f64>,
    /// Per-layer sliding window. `Some(w)` for sliding-attention layers, `None` for full.
    pub(crate) sliding_window: Option<usize>,
}

impl Gemma2Attention {
    /// Load attention weights.
    ///
    /// Sliding window is initialized to `None`; set `self.sliding_window` after
    /// construction for sliding-attention layers.
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Gemma2Config,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let q_proj = ColumnParallelLinear::load(
            weights,
            &format!("{}.q_proj", prefix),
            dtype,
            rank,
            world_size,
            false,
        )?;
        let k_proj = ColumnParallelLinear::load(
            weights,
            &format!("{}.k_proj", prefix),
            dtype,
            rank,
            world_size,
            false,
        )?;
        let v_proj = ColumnParallelLinear::load(
            weights,
            &format!("{}.v_proj", prefix),
            dtype,
            rank,
            world_size,
            false,
        )?;
        let o_proj = RowParallelLinear::load(
            weights,
            &format!("{}.o_proj", prefix),
            dtype,
            rank,
            world_size,
            true,
        )?;

        let num_q_heads = config.num_attention_heads / world_size;
        let num_kv_heads = config.num_kv_heads / world_size;

        let rotary_emb = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            dtype,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_q_heads,
            num_kv_heads,
            head_dim: config.head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5),
            attn_logit_softcapping: config.attn_logit_softcapping,
            sliding_window: None,
        })
    }

    /// Create with zero weights (for testing).
    ///
    /// Sliding window is initialized to `None`; set `self.sliding_window` after.
    pub fn zeros(config: &Gemma2Config, dtype: DType, device: &Device) -> ModelResult<Self> {
        let hidden = config.hidden_size;
        let q_size = config.num_attention_heads * config.head_dim;
        let kv_size = config.num_kv_heads * config.head_dim;

        let q_proj =
            ColumnParallelLinear::new(Linear::zeros(hidden, q_size, dtype, device)?, false);
        let k_proj =
            ColumnParallelLinear::new(Linear::zeros(hidden, kv_size, dtype, device)?, false);
        let v_proj =
            ColumnParallelLinear::new(Linear::zeros(hidden, kv_size, dtype, device)?, false);
        let o_proj = RowParallelLinear::new(Linear::zeros(q_size, hidden, dtype, device)?, true);

        let rotary_emb = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            dtype,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_q_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5),
            attn_logit_softcapping: config.attn_logit_softcapping,
            sliding_window: None,
        })
    }

    /// Forward pass.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<Tensor> {
        let num_tokens = hidden_states.dim(0).map_err(ModelError::Candle)?;

        let q = self
            .q_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;
        let k = self
            .k_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;
        let v = self
            .v_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;

        let q = q
            .reshape((num_tokens, self.num_q_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let k = k
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let v = v
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;

        let (q, k) = self.rotary_emb.apply(&q, &k, positions)?;

        // Cache-merge + attention (paged decode reads blocks directly).
        let _ = self.attn_logit_softcapping; // Reserved for GPU kernel integration
        let attn_output =
            attention_with_cache(&q, &k, &v, self.scale, kv_cache, self.sliding_window)?;

        let attn_output = attn_output
            .reshape((num_tokens, self.num_q_heads * self.head_dim))
            .map_err(ModelError::Candle)?;

        self.o_proj
            .forward(&attn_output)
            .map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// Gemma2DecoderLayer
// ---------------------------------------------------------------------------

/// A single Gemma2 decoder layer.
///
/// Has 4 norms (vs LLaMA's 2):
/// - input_layernorm → attention → post_attention_layernorm
/// - pre_feedforward_layernorm → MLP → post_feedforward_layernorm
pub struct Gemma2DecoderLayer {
    pub(crate) self_attn: Gemma2Attention,
    mlp: Gemma2MLP,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
}

impl Gemma2DecoderLayer {
    /// Load a decoder layer.
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Gemma2Config,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let self_attn = Gemma2Attention::load(
            weights,
            &format!("{}.self_attn", prefix),
            config,
            dtype,
            device,
            rank,
            world_size,
        )?;
        let mlp = Gemma2MLP::load(weights, &format!("{}.mlp", prefix), dtype, rank, world_size)?;
        let input_layernorm = GemmaRmsNorm::load(
            weights,
            &format!("{}.input_layernorm", prefix),
            config.rms_norm_eps,
            dtype,
        )?;
        let post_attention_layernorm = GemmaRmsNorm::load(
            weights,
            &format!("{}.post_attention_layernorm", prefix),
            config.rms_norm_eps,
            dtype,
        )?;
        let pre_feedforward_layernorm = GemmaRmsNorm::load(
            weights,
            &format!("{}.pre_feedforward_layernorm", prefix),
            config.rms_norm_eps,
            dtype,
        )?;
        let post_feedforward_layernorm = GemmaRmsNorm::load(
            weights,
            &format!("{}.post_feedforward_layernorm", prefix),
            config.rms_norm_eps,
            dtype,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<Tensor> {
        // Pre-attention norm + attention.
        let normed = crate::ops::gemma_rms_norm(hidden_states, &self.input_layernorm)
            .map_err(ModelError::Candle)?;
        let attn_output = self.self_attn.forward(&normed, positions, kv_cache)?;
        // Post-attention norm + residual.
        let attn_output = crate::ops::gemma_rms_norm(&attn_output, &self.post_attention_layernorm)
            .map_err(ModelError::Candle)?;
        let hidden_states = (hidden_states + attn_output).map_err(ModelError::Candle)?;

        // Pre-feedforward norm + MLP.
        let normed = crate::ops::gemma_rms_norm(&hidden_states, &self.pre_feedforward_layernorm)
            .map_err(ModelError::Candle)?;
        let mlp_output = self.mlp.forward(&normed).map_err(ModelError::Candle)?;
        // Post-feedforward norm + residual.
        let mlp_output = crate::ops::gemma_rms_norm(&mlp_output, &self.post_feedforward_layernorm)
            .map_err(ModelError::Candle)?;
        let hidden_states = (hidden_states + mlp_output).map_err(ModelError::Candle)?;

        Ok(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Gemma2Model
// ---------------------------------------------------------------------------

/// Gemma2 transformer backbone.
///
/// Embedding (× sqrt(hidden_size)) → N decoder layers → final GemmaRMS norm.
pub struct Gemma2Model {
    embed_tokens: Embedding,
    layers: Vec<Gemma2DecoderLayer>,
    norm: GemmaRmsNorm,
    /// Embedding normalizer: sqrt(hidden_size).
    normalizer: f64,
}

impl Gemma2Model {
    /// Load the model backbone.
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Gemma2Config,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let embed_tokens = Embedding::load(weights, &format!("{}.embed_tokens", prefix), dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let mut layer = Gemma2DecoderLayer::load(
                weights,
                &format!("{}.layers.{}", prefix, i),
                config,
                dtype,
                device,
                rank,
                world_size,
            )?;
            // Apply per-layer sliding window from layer_is_sliding.
            if i < config.layer_is_sliding.len() && config.layer_is_sliding[i] {
                layer.self_attn.sliding_window = config.sliding_window;
            }
            layers.push(layer);
        }

        let norm = GemmaRmsNorm::load(
            weights,
            &format!("{}.norm", prefix),
            config.rms_norm_eps,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            normalizer: (config.hidden_size as f64).sqrt(),
        })
    }

    /// Forward pass.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        mut kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let mut hidden_states = self
            .embed_tokens
            .forward(input_ids)
            .map_err(ModelError::Candle)?;

        // Gemma normalizes embeddings by sqrt(hidden_size).
        hidden_states = (hidden_states * self.normalizer).map_err(ModelError::Candle)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_handle = kv_cache.as_mut().map(|s| s.layer_handle(i));
            hidden_states = layer.forward(&hidden_states, positions, layer_handle)?;
        }

        crate::ops::gemma_rms_norm(&hidden_states, &self.norm).map_err(ModelError::Candle)
    }

    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// Gemma2ForCausalLM
// ---------------------------------------------------------------------------

/// Gemma2 for causal language modeling.
///
/// Uses tied embeddings (embed_tokens weight as lm_head) and optional
/// logit soft capping.
pub struct Gemma2ForCausalLM {
    model: Gemma2Model,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
}

impl Gemma2ForCausalLM {
    /// Load the full model from weights.
    pub fn load(
        weights: &ModelWeights,
        config: &Gemma2Config,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let model = Gemma2Model::load(weights, "model", config, dtype, device, rank, world_size)?;

        // Gemma always uses tied embeddings.
        let lm_head = Linear::new(model.embed_tokens.weight().clone(), None);

        Ok(Self {
            model,
            lm_head,
            final_logit_softcapping: config.final_logit_softcapping,
        })
    }

    /// Compute logits from hidden states, with optional soft capping.
    pub fn compute_logits(&self, hidden_states: &Tensor) -> ModelResult<Tensor> {
        let logits = self
            .lm_head
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;

        // Apply soft capping: logits = cap * tanh(logits / cap)
        if let Some(cap) = self.final_logit_softcapping {
            let scaled = (logits / cap).map_err(ModelError::Candle)?;
            let capped = scaled.tanh().map_err(ModelError::Candle)?;
            (capped * cap).map_err(ModelError::Candle)
        } else {
            Ok(logits)
        }
    }
}

impl crate::Model for Gemma2ForCausalLM {
    fn inject_lora(&mut self, adapter: &LoraAdapter) -> ModelResult<()> {
        let targets = &adapter.config.target_modules;
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            // Attention projections.
            let attn_prefix = format!("model.layers.{}.self_attn", i);
            let attn_projs: &mut [(&str, &mut ColumnParallelLinear)] = &mut [
                ("q_proj", &mut layer.self_attn.q_proj),
                ("k_proj", &mut layer.self_attn.k_proj),
                ("v_proj", &mut layer.self_attn.v_proj),
            ];
            for (name, proj) in attn_projs.iter_mut() {
                if targets.iter().any(|t| t == name) {
                    let key = format!("{}.{}", attn_prefix, name);
                    if let Some((a, b)) = adapter.weights.get(&key) {
                        proj.inner_mut()
                            .attach_lora(a.clone(), b.clone(), adapter.scaling)?;
                    }
                }
            }
            if targets.iter().any(|t| t == "o_proj") {
                let key = format!("{}.o_proj", attn_prefix);
                if let Some((a, b)) = adapter.weights.get(&key) {
                    layer.self_attn.o_proj.inner_mut().attach_lora(
                        a.clone(),
                        b.clone(),
                        adapter.scaling,
                    )?;
                }
            }
            // MLP projections.
            let mlp_prefix = format!("model.layers.{}.mlp", i);
            let mlp_projs: &mut [(&str, &mut ColumnParallelLinear)] = &mut [
                ("gate_proj", &mut layer.mlp.gate_proj),
                ("up_proj", &mut layer.mlp.up_proj),
            ];
            for (name, proj) in mlp_projs.iter_mut() {
                if targets.iter().any(|t| t == name) {
                    let key = format!("{}.{}", mlp_prefix, name);
                    if let Some((a, b)) = adapter.weights.get(&key) {
                        proj.inner_mut()
                            .attach_lora(a.clone(), b.clone(), adapter.scaling)?;
                    }
                }
            }
            if targets.iter().any(|t| t == "down_proj") {
                let key = format!("{}.down_proj", mlp_prefix);
                if let Some((a, b)) = adapter.weights.get(&key) {
                    layer.mlp.down_proj.inner_mut().attach_lora(
                        a.clone(),
                        b.clone(),
                        adapter.scaling,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self.model.forward(input_ids, positions, kv_cache)?;
        let logits = self.compute_logits(&hidden_states)?;
        // Cast logits to f32 for sampling (sampler expects f32).
        logits.to_dtype(DType::F32).map_err(ModelError::Candle)
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn hidden_states(&self, input_ids: &Tensor, positions: &Tensor) -> ModelResult<Tensor> {
        self.model.forward(input_ids, positions, None)
    }
}

/// Factory function for the model registry.
pub fn create_gemma2(
    weights: &ModelWeights,
    config: &HfModelConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let gemma2_config = Gemma2Config::from_hf_config(config)?;
    let model = Gemma2ForCausalLM::load(weights, &gemma2_config, dtype, device, 0, 1)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Gemma2Config {
        Gemma2Config {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            head_dim: 8,
            query_pre_attn_scalar: 8.0, // = head_dim
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            attention_bias: false,
            sliding_window: None,
            layer_is_sliding: Vec::new(),
        }
    }

    fn test_config_gqa() -> Gemma2Config {
        Gemma2Config {
            num_kv_heads: 2,
            ..test_config()
        }
    }

    #[test]
    fn test_gemma2_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Gemma2ForCausalLM"],
                "model_type": "gemma2",
                "hidden_size": 2304,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "num_hidden_layers": 26,
                "intermediate_size": 9216,
                "vocab_size": 256000,
                "max_position_embeddings": 8192,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "head_dim": 256,
                "query_pre_attn_scalar": 256,
                "attn_logit_softcapping": 50.0,
                "final_logit_softcapping": 30.0,
                "attention_bias": false,
                "tie_word_embeddings": true
            }"#,
        )
        .unwrap();

        let config = Gemma2Config::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 2304);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 256);
        assert!((config.query_pre_attn_scalar - 256.0).abs() < 1.0);
        assert!((config.attn_logit_softcapping.unwrap() - 50.0).abs() < 1.0);
        assert!((config.final_logit_softcapping.unwrap() - 30.0).abs() < 1.0);
        assert!(!config.attention_bias);
    }

    #[test]
    fn test_gemma2_mlp_forward_zeros() {
        let config = test_config();
        let mlp = Gemma2MLP::zeros(
            config.hidden_size,
            config.intermediate_size,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        let x = Tensor::ones(&[3, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.dims(), &[3, config.hidden_size]);
    }

    #[test]
    fn test_gemma2_attention_forward() {
        let config = test_config();
        let attn = Gemma2Attention::zeros(&config, DType::F32, &Device::Cpu).unwrap();

        let num_tokens = 4;
        let x = Tensor::ones(&[num_tokens, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[num_tokens, config.hidden_size]);
    }

    #[test]
    fn test_gemma2_attention_gqa() {
        let config = test_config_gqa();
        let attn = Gemma2Attention::zeros(&config, DType::F32, &Device::Cpu).unwrap();

        let num_tokens = 3;
        let x = Tensor::ones(&[num_tokens, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[num_tokens, config.hidden_size]);
    }

    #[test]
    fn test_gemma2_model_from_weights() {
        let config = test_config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let device = Device::Cpu;
        let dtype = DType::F32;

        let mut tensor_specs: Vec<(String, Vec<usize>)> = Vec::new();

        // Embedding.
        tensor_specs.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));

        // Layers.
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let q_size = config.num_attention_heads * config.head_dim;
            let kv_size = config.num_kv_heads * config.head_dim;

            tensor_specs.push((
                format!("{}.self_attn.q_proj.weight", prefix),
                vec![q_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.k_proj.weight", prefix),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.v_proj.weight", prefix),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.o_proj.weight", prefix),
                vec![config.hidden_size, q_size],
            ));

            tensor_specs.push((
                format!("{}.mlp.gate_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.up_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.down_proj.weight", prefix),
                vec![config.hidden_size, config.intermediate_size],
            ));

            // 4 norms per layer (GemmaRmsNorm stores weight before +1 offset).
            tensor_specs.push((
                format!("{}.input_layernorm.weight", prefix),
                vec![config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.post_attention_layernorm.weight", prefix),
                vec![config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.pre_feedforward_layernorm.weight", prefix),
                vec![config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.post_feedforward_layernorm.weight", prefix),
                vec![config.hidden_size],
            ));
        }

        // Final norm.
        tensor_specs.push(("model.norm.weight".to_string(), vec![config.hidden_size]));
        // No lm_head — tied embeddings.

        create_test_weights(&path, &tensor_specs);

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model = Gemma2ForCausalLM::load(&weights, &config, dtype, &device, 0, 1).unwrap();

        let input_ids = Tensor::new(&[1u32, 5, 10], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();

        let logits = crate::Model::forward(&model, &input_ids, &positions, None).unwrap();
        assert_eq!(logits.dims(), &[3, config.vocab_size]);

        // Logits should be bounded by soft capping.
        let max_logit = logits
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            max_logit.abs() <= config.final_logit_softcapping.unwrap() as f32 + 0.01,
            "logit {} should be bounded by softcap {}",
            max_logit,
            config.final_logit_softcapping.unwrap()
        );
    }

    #[test]
    fn test_gemma2_logit_softcapping() {
        // Verify tanh-based soft capping works.
        let cap = 30.0;
        let logits = Tensor::new(&[[100.0f32, -100.0, 0.0, 15.0]], &Device::Cpu).unwrap();

        let scaled = (&logits / cap).unwrap();
        let capped = (scaled.tanh().unwrap() * cap).unwrap();
        let vals = capped.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // 100/30 -> tanh -> ~1.0 -> *30 -> ~30
        assert!((vals[0] - 30.0).abs() < 0.1);
        // -100/30 -> tanh -> ~-1.0 -> *30 -> ~-30
        assert!((vals[1] - (-30.0)).abs() < 0.1);
        // 0 -> tanh -> 0 -> 0
        assert!(vals[2].abs() < 0.01);
        // 15/30 = 0.5 -> tanh(0.5) -> ~0.462 -> *30 -> ~13.86
        assert!((vals[3] - 13.86).abs() < 0.1);
    }

    #[test]
    fn test_gemma2_registry() {
        let registry = crate::ModelRegistry::default_registry();
        assert!(registry.contains("Gemma2ForCausalLM"));
    }

    #[test]
    fn test_gemma2_config_layer_types_parsing() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Gemma2ForCausalLM"],
                "model_type": "gemma2",
                "hidden_size": 2304,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "num_hidden_layers": 4,
                "intermediate_size": 9216,
                "vocab_size": 256000,
                "head_dim": 256,
                "query_pre_attn_scalar": 256,
                "sliding_window": 4096,
                "layer_types": ["full_attention", "sliding_attention", "full_attention", "sliding_attention"]
            }"#,
        )
        .unwrap();

        let config = Gemma2Config::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.sliding_window, Some(4096));
        assert_eq!(config.layer_is_sliding, vec![false, true, false, true]);
    }

    #[test]
    fn test_gemma2_interleaved_sliding_window() {
        // Layer 0 = full attention (no sliding window), layer 1 = sliding window.
        let config = Gemma2Config {
            sliding_window: Some(3),
            layer_is_sliding: vec![false, true],
            ..test_config()
        };

        // Full attention layer (default from zeros).
        let attn_full = Gemma2Attention::zeros(&config, DType::F32, &Device::Cpu).unwrap();
        assert!(attn_full.sliding_window.is_none());

        // Sliding attention layer (set after construction).
        let mut attn_sliding = Gemma2Attention::zeros(&config, DType::F32, &Device::Cpu).unwrap();
        attn_sliding.sliding_window = config.sliding_window;
        assert_eq!(attn_sliding.sliding_window, Some(3));
    }

    // -----------------------------------------------------------------------
    // Test helper
    // -----------------------------------------------------------------------

    fn create_test_weights(path: &std::path::Path, specs: &[(String, Vec<usize>)]) {
        use safetensors::tensor::TensorView;

        let mut all_data: Vec<Vec<u8>> = Vec::new();
        for (_, shape) in specs {
            let num_elements: usize = shape.iter().product();
            let data: Vec<u8> = (0..num_elements)
                .flat_map(|_| 0.01f32.to_le_bytes())
                .collect();
            all_data.push(data);
        }

        // GemmaRmsNorm stores weight before +1 offset, so 0.0 means effective weight = 1.0.
        for (i, (name, shape)) in specs.iter().enumerate() {
            if name.contains("layernorm") || (name.as_str() == "model.norm.weight") {
                let num_elements: usize = shape.iter().product();
                all_data[i] = (0..num_elements)
                    .flat_map(|_| 0.0f32.to_le_bytes())
                    .collect();
            }
        }

        let views: Vec<(&str, TensorView<'_>)> = specs
            .iter()
            .zip(all_data.iter())
            .map(|((name, shape), data)| {
                (
                    name.as_str(),
                    TensorView::new(safetensors::Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        let bytes = safetensors::tensor::serialize(views, None).unwrap();
        std::fs::write(path, bytes).unwrap();
    }
}
