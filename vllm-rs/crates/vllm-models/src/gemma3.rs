// SPDX-License-Identifier: Apache-2.0
//! Gemma 3 text-only model architecture.
//!
//! Implements:
//! - `Gemma3ForCausalLM` — top-level model with tied embeddings
//! - `Gemma3Model` — transformer backbone with embedding normalization
//! - `Gemma3DecoderLayer` — single transformer layer with 4 norms
//! - `Gemma3Attention` — attention with per-head QK norms and per-layer RoPE theta
//! - `Gemma3MLP` — GELU-gated feed-forward network
//!
//! Key differences from Gemma 2:
//! - No `attn_logit_softcapping` (removed)
//! - No `final_logit_softcapping` (removed)
//! - Per-head `q_norm` + `k_norm` (GemmaRmsNorm with +1 offset)
//! - `sliding_window_pattern: N` instead of explicit `layer_types` array
//! - Per-layer RoPE theta: `rope_theta` (global) vs `rope_local_base_freq` (sliding)
//! - Everything else identical: 4 norms, GemmaRmsNorm, GELU tanh, embed*sqrt(H), tied embeddings
//!
//! Port of: `vllm/model_executor/models/gemma3.py`

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{
    ColumnParallelLinear, Embedding, GemmaRmsNorm, Linear, RotaryEmbedding, RowParallelLinear,
};
use vllm_model::lora::LoraAdapter;
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::attention::attention_with_cache;

// ---------------------------------------------------------------------------
// Gemma3Config
// ---------------------------------------------------------------------------

/// Parsed configuration for a Gemma 3 model.
#[derive(Debug, Clone)]
pub struct Gemma3Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// RoPE theta for local/sliding-window layers.
    pub rope_local_base_freq: f64,
    pub head_dim: usize,
    /// Scaling factor for query before attention (replaces 1/sqrt(head_dim)).
    pub query_pre_attn_scalar: f64,
    /// Optional soft cap for attention logits (None for Gemma3).
    pub attn_logit_softcapping: Option<f64>,
    /// Optional soft cap for final logits (None for Gemma3).
    pub final_logit_softcapping: Option<f64>,
    /// Whether attention projections use bias.
    pub attention_bias: bool,
    /// Sliding window size for local layers. None if not set.
    pub sliding_window: Option<usize>,
    /// Per-layer attention type: `true` = sliding/local, `false` = global/full.
    pub layer_is_sliding: Vec<bool>,
}

impl Gemma3Config {
    /// Parse from a HuggingFace config.json.
    ///
    /// Defaults match the transformers `Gemma3TextConfig` class so that
    /// multimodal configs (which store only overrides in `text_config`)
    /// are handled correctly.
    pub fn from_hf_config(config: &HfModelConfig) -> ModelResult<Self> {
        let hidden_size = config
            .hidden_size
            .ok_or_else(|| ModelError::Other("missing hidden_size".into()))?;
        // Gemma3TextConfig default: num_attention_heads=8, num_key_value_heads=4.
        let num_attention_heads = config.num_attention_heads.unwrap_or(8);
        // Gemma 3 default head_dim is 256, which may differ from
        // hidden_size / num_attention_heads (e.g. 3840/16 = 240 for 12B).
        // Use the raw field, not the computed fallback.
        let head_dim = config.head_dim.unwrap_or(256);

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

        let rope_theta = config.rope_theta.unwrap_or(1_000_000.0);

        let rope_local_base_freq = config
            .extra
            .get("rope_local_base_freq")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);

        let num_hidden_layers = config
            .num_hidden_layers
            .ok_or_else(|| ModelError::Other("missing num_hidden_layers".into()))?;

        // Gemma3 uses `sliding_window_pattern` to determine which layers are
        // local (sliding) vs global (full). Every Nth layer is global, the rest
        // are local: is_sliding(i) = (i + 1) % pattern != 0
        // Falls back to `layer_types` array (Gemma2 format) if present.
        // Default pattern is 6 when sliding_window is set (matches transformers).
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
                .ok_or_else(|| ModelError::Other("missing intermediate_size".into()))?,
            vocab_size: config.vocab_size.unwrap_or(262144),
            max_position_embeddings: config.max_position_embeddings.unwrap_or(131072),
            rms_norm_eps: config.norm_eps(),
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

    /// Returns the RoPE theta for a given layer based on whether it uses sliding attention.
    fn rope_theta_for_layer(&self, layer_idx: usize) -> f64 {
        if layer_idx < self.layer_is_sliding.len() && self.layer_is_sliding[layer_idx] {
            self.rope_local_base_freq
        } else {
            self.rope_theta
        }
    }
}

// ---------------------------------------------------------------------------
// Gemma3MLP
// ---------------------------------------------------------------------------

/// Gemma3 MLP (GELU-gated feed-forward network).
///
/// Forward: gate_proj(x) -> GELU(tanh) -> * up_proj(x) -> down_proj
pub struct Gemma3MLP {
    gate_proj: ColumnParallelLinear,
    up_proj: ColumnParallelLinear,
    down_proj: RowParallelLinear,
}

impl Gemma3MLP {
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

impl Module for Gemma3MLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        // GELU(tanh)(gate) * up
        let activated = crate::ops::gelu_and_mul(&gate, &up)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// Gemma3Attention
// ---------------------------------------------------------------------------

/// Gemma3 multi-head attention with per-head QK norms and per-layer RoPE theta.
pub struct Gemma3Attention {
    q_proj: ColumnParallelLinear,
    k_proj: ColumnParallelLinear,
    v_proj: ColumnParallelLinear,
    o_proj: RowParallelLinear,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    rotary_emb: RotaryEmbedding,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    /// Per-layer sliding window. `Some(w)` for sliding layers, `None` for global.
    pub(crate) sliding_window: Option<usize>,
}

impl Gemma3Attention {
    /// Load attention weights.
    ///
    /// `layer_idx` is used to determine RoPE theta (global vs local).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Gemma3Config,
        layer_idx: usize,
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

        let q_norm = GemmaRmsNorm::load(
            weights,
            &format!("{}.q_norm", prefix),
            config.rms_norm_eps,
            dtype,
        )?;
        let k_norm = GemmaRmsNorm::load(
            weights,
            &format!("{}.k_norm", prefix),
            config.rms_norm_eps,
            dtype,
        )?;

        let num_q_heads = config.num_attention_heads / world_size;
        let num_kv_heads = config.num_kv_heads / world_size;

        let rope_theta = config.rope_theta_for_layer(layer_idx);
        let rotary_emb = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            rope_theta,
            dtype,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            num_q_heads,
            num_kv_heads,
            head_dim: config.head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5),
            sliding_window: None,
        })
    }

    /// Create with zero weights (for testing).
    pub fn zeros(
        config: &Gemma3Config,
        layer_idx: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
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

        // Zero GemmaRmsNorm: weight=0 → effective weight=(0+1)=1
        let q_norm_weight =
            Tensor::zeros(config.head_dim, dtype, device).map_err(ModelError::Candle)?;
        let k_norm_weight =
            Tensor::zeros(config.head_dim, dtype, device).map_err(ModelError::Candle)?;
        let q_norm =
            GemmaRmsNorm::new(q_norm_weight, config.rms_norm_eps).map_err(ModelError::Candle)?;
        let k_norm =
            GemmaRmsNorm::new(k_norm_weight, config.rms_norm_eps).map_err(ModelError::Candle)?;

        let rope_theta = config.rope_theta_for_layer(layer_idx);
        let rotary_emb = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            rope_theta,
            dtype,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            num_q_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5),
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

        // Reshape to [tokens, heads, head_dim].
        let q = q
            .reshape((num_tokens, self.num_q_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let k = k
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let v = v
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;

        // Fused per-head QK norm + RoPE: on CUDA this is a single kernel,
        // on CPU it falls back to separate norm + rotation.
        let (q, k) = crate::ops::qk_norm_and_rope(
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.rotary_emb.cos_cache(),
            self.rotary_emb.sin_cache(),
            positions,
        )
        .map_err(ModelError::Candle)?;

        // Cache-merge + attention
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
// Gemma3DecoderLayer
// ---------------------------------------------------------------------------

/// A single Gemma3 decoder layer.
///
/// Has 4 norms (same structure as Gemma2):
/// - input_layernorm -> attention -> post_attention_layernorm
/// - pre_feedforward_layernorm -> MLP -> post_feedforward_layernorm
pub struct Gemma3DecoderLayer {
    pub(crate) self_attn: Gemma3Attention,
    mlp: Gemma3MLP,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
}

impl Gemma3DecoderLayer {
    /// Load a decoder layer.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Gemma3Config,
        layer_idx: usize,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let self_attn = Gemma3Attention::load(
            weights,
            &format!("{}.self_attn", prefix),
            config,
            layer_idx,
            dtype,
            device,
            rank,
            world_size,
        )?;
        let mlp = Gemma3MLP::load(weights, &format!("{}.mlp", prefix), dtype, rank, world_size)?;
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
// Gemma3Model
// ---------------------------------------------------------------------------

/// Gemma3 transformer backbone.
///
/// Embedding (* sqrt(hidden_size)) -> N decoder layers -> final GemmaRMS norm.
pub struct Gemma3Model {
    pub(crate) embed_tokens: Embedding,
    layers: Vec<Gemma3DecoderLayer>,
    norm: GemmaRmsNorm,
    /// Embedding normalizer: sqrt(hidden_size).
    normalizer: f64,
}

impl Gemma3Model {
    /// Load the model backbone.
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Gemma3Config,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let embed_tokens = Embedding::load(weights, &format!("{}.embed_tokens", prefix), dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let mut layer = Gemma3DecoderLayer::load(
                weights,
                &format!("{}.layers.{}", prefix, i),
                config,
                i,
                dtype,
                device,
                rank,
                world_size,
            )?;
            // Apply per-layer sliding window.
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

    /// Embed token IDs and scale by sqrt(hidden_size).
    pub fn embed(&self, input_ids: &Tensor) -> ModelResult<Tensor> {
        let hidden_states = self
            .embed_tokens
            .forward(input_ids)
            .map_err(ModelError::Candle)?;
        (hidden_states * self.normalizer).map_err(ModelError::Candle)
    }

    /// Run the transformer backbone on pre-computed embeddings.
    pub fn backbone(
        &self,
        mut hidden_states: Tensor,
        positions: &Tensor,
        mut kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_handle = kv_cache.as_mut().map(|s| s.layer_handle(i));
            hidden_states = layer.forward(&hidden_states, positions, layer_handle)?;
        }

        crate::ops::gemma_rms_norm(&hidden_states, &self.norm).map_err(ModelError::Candle)
    }

    /// Forward pass (embed + backbone).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self.embed(input_ids)?;
        self.backbone(hidden_states, positions, kv_cache)
    }

    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// Gemma3ForCausalLM
// ---------------------------------------------------------------------------

/// Gemma3 for causal language modeling.
///
/// Uses tied embeddings (embed_tokens weight as lm_head) and optional
/// logit soft capping (None for Gemma3, kept for robustness).
pub struct Gemma3ForCausalLM {
    pub(crate) model: Gemma3Model,
    pub(crate) lm_head: Linear,
    pub(crate) final_logit_softcapping: Option<f64>,
}

impl Gemma3ForCausalLM {
    /// Load the full model from weights.
    pub fn load(
        weights: &ModelWeights,
        config: &Gemma3Config,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let model = Gemma3Model::load(weights, "model", config, dtype, device, rank, world_size)?;

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

        if let Some(cap) = self.final_logit_softcapping {
            let scaled = (logits / cap).map_err(ModelError::Candle)?;
            let capped = scaled.tanh().map_err(ModelError::Candle)?;
            (capped * cap).map_err(ModelError::Candle)
        } else {
            Ok(logits)
        }
    }
}

impl crate::Model for Gemma3ForCausalLM {
    fn inject_lora(&mut self, adapter: &LoraAdapter) -> ModelResult<()> {
        let targets = &adapter.config.target_modules;
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
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

    fn forward_embeds(
        &self,
        inputs_embeds: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self
            .model
            .backbone(inputs_embeds.clone(), positions, kv_cache)?;
        let logits = self.compute_logits(&hidden_states)?;
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
pub fn create_gemma3(
    weights: &ModelWeights,
    config: &HfModelConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let gemma3_config = Gemma3Config::from_hf_config(config)?;
    let model = Gemma3ForCausalLM::load(weights, &gemma3_config, dtype, device, 0, 1)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Gemma3Config {
        Gemma3Config {
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
            // Layer 0 = sliding, layer 1 = global (pattern=2)
            layer_is_sliding: vec![true, false],
        }
    }

    fn test_config_gqa() -> Gemma3Config {
        Gemma3Config {
            num_kv_heads: 2,
            ..test_config()
        }
    }

    #[test]
    fn test_gemma3_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Gemma3ForCausalLM"],
                "model_type": "gemma3",
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
                "attention_bias": false,
                "tie_word_embeddings": true
            }"#,
        )
        .unwrap();

        let config = Gemma3Config::from_hf_config(&hf_config).unwrap();
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
    }

    #[test]
    fn test_gemma3_config_from_multimodal_text_config_12b() {
        // Gemma 3 12B multimodal text_config — specifies num_attention_heads
        // but omits vocab_size, head_dim, etc.
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "hidden_size": 3840,
                "intermediate_size": 15360,
                "model_type": "gemma3_text",
                "num_attention_heads": 16,
                "num_hidden_layers": 48,
                "num_key_value_heads": 8,
                "sliding_window": 1024
            }"#,
        )
        .unwrap();

        let config = Gemma3Config::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 3840);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.num_hidden_layers, 48);
        assert_eq!(config.intermediate_size, 15360);
        // Defaults from transformers Gemma3TextConfig:
        assert_eq!(config.vocab_size, 262144);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.max_position_embeddings, 131072);
        assert!((config.rope_theta - 1_000_000.0).abs() < 1.0);
        assert_eq!(config.sliding_window, Some(1024));
        // Default sliding_window_pattern=6 inferred from sliding_window presence.
        assert_eq!(config.layer_is_sliding.len(), 48);
        assert!(config.layer_is_sliding[0]); // layer 0: sliding
        assert!(config.layer_is_sliding[4]); // layer 4: sliding
        assert!(!config.layer_is_sliding[5]); // layer 5: global (6th)
        assert!(!config.layer_is_sliding[11]); // layer 11: global (12th)
    }

    #[test]
    fn test_gemma3_config_from_multimodal_text_config_4b() {
        // Gemma 3 4B multimodal text_config — omits num_attention_heads,
        // num_key_value_heads, vocab_size, head_dim, and everything else
        // that matches the transformers Gemma3TextConfig defaults.
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "hidden_size": 2560,
                "intermediate_size": 10240,
                "model_type": "gemma3_text",
                "num_hidden_layers": 34,
                "sliding_window": 1024
            }"#,
        )
        .unwrap();

        let config = Gemma3Config::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_hidden_layers, 34);
        assert_eq!(config.intermediate_size, 10240);
        // All of these come from Gemma3TextConfig defaults:
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.vocab_size, 262144);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.max_position_embeddings, 131072);
        assert!((config.rope_theta - 1_000_000.0).abs() < 1.0);
        assert_eq!(config.sliding_window, Some(1024));
        assert_eq!(config.layer_is_sliding.len(), 34);
    }

    #[test]
    fn test_gemma3_sliding_window_pattern() {
        // sliding_window_pattern=2 means every 2nd layer (1-indexed) is global.
        // Layer 0: (0+1)%2 = 1 != 0 -> sliding
        // Layer 1: (1+1)%2 = 0 == 0 -> global
        // ...
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Gemma3ForCausalLM"],
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_hidden_layers": 6,
                "intermediate_size": 64,
                "vocab_size": 100,
                "head_dim": 8,
                "query_pre_attn_scalar": 8,
                "sliding_window_pattern": 2
            }"#,
        )
        .unwrap();

        let config = Gemma3Config::from_hf_config(&hf_config).unwrap();
        assert_eq!(
            config.layer_is_sliding,
            vec![true, false, true, false, true, false]
        );
    }

    #[test]
    fn test_gemma3_sliding_window_pattern_3() {
        // sliding_window_pattern=3: every 3rd layer is global.
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Gemma3ForCausalLM"],
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_hidden_layers": 6,
                "intermediate_size": 64,
                "vocab_size": 100,
                "head_dim": 8,
                "query_pre_attn_scalar": 8,
                "sliding_window_pattern": 3
            }"#,
        )
        .unwrap();

        let config = Gemma3Config::from_hf_config(&hf_config).unwrap();
        // (0+1)%3=1!=0 -> sliding, (1+1)%3=2!=0 -> sliding, (2+1)%3=0 -> global
        // (3+1)%3=1!=0 -> sliding, (4+1)%3=2!=0 -> sliding, (5+1)%3=0 -> global
        assert_eq!(
            config.layer_is_sliding,
            vec![true, true, false, true, true, false]
        );
    }

    #[test]
    fn test_gemma3_rope_theta_per_layer() {
        let config = test_config();
        // Layer 0 = sliding -> local base freq
        assert!((config.rope_theta_for_layer(0) - 10000.0).abs() < 1.0);
        // Layer 1 = global -> global theta
        assert!((config.rope_theta_for_layer(1) - 1000000.0).abs() < 1.0);
    }

    #[test]
    fn test_gemma3_mlp_forward_zeros() {
        let config = test_config();
        let mlp = Gemma3MLP::zeros(
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
    fn test_gemma3_attention_forward() {
        let config = test_config();
        let attn = Gemma3Attention::zeros(&config, 0, DType::F32, &Device::Cpu).unwrap();

        let num_tokens = 4;
        let x = Tensor::ones(&[num_tokens, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[num_tokens, config.hidden_size]);
    }

    #[test]
    fn test_gemma3_attention_gqa() {
        let config = test_config_gqa();
        let attn = Gemma3Attention::zeros(&config, 0, DType::F32, &Device::Cpu).unwrap();

        let num_tokens = 3;
        let x = Tensor::ones(&[num_tokens, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[num_tokens, config.hidden_size]);
    }

    #[test]
    fn test_gemma3_model_from_weights() {
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

            // Per-head QK norms.
            tensor_specs.push((
                format!("{}.self_attn.q_norm.weight", prefix),
                vec![config.head_dim],
            ));
            tensor_specs.push((
                format!("{}.self_attn.k_norm.weight", prefix),
                vec![config.head_dim],
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

            // 4 norms per layer.
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

        create_test_weights(&path, &tensor_specs);

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model = Gemma3ForCausalLM::load(&weights, &config, dtype, &device, 0, 1).unwrap();

        let input_ids = Tensor::new(&[1u32, 5, 10], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();

        let logits = crate::Model::forward(&model, &input_ids, &positions, None).unwrap();
        assert_eq!(logits.dims(), &[3, config.vocab_size]);

        // No softcapping — logits should NOT be bounded by any cap.
        // (They should just be whatever the model computes.)
    }

    #[test]
    fn test_gemma3_registry() {
        let registry = crate::ModelRegistry::default_registry();
        assert!(registry.contains("Gemma3ForCausalLM"));
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
            if name.contains("layernorm")
                || name.contains("_norm.weight")
                || (name.as_str() == "model.norm.weight")
            {
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
