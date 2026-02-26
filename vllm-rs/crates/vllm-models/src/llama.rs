// SPDX-License-Identifier: Apache-2.0
//! LLaMA model architecture.
//!
//! Implements:
//! - `LlamaForCausalLM` — top-level model with lm_head and logits
//! - `LlamaModel` — transformer backbone (embed + layers + final norm)
//! - `LlamaDecoderLayer` — single transformer layer (attention + MLP + norms)
//! - `LlamaAttention` — multi-head attention with RoPE and GQA
//! - `LlamaMLP` — SiLU-gated feed-forward network
//!
//! Also covers Mistral, which uses the same architecture.
//!
//! Port of: `vllm/model_executor/models/llama.py`

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{
    ColumnParallelLinear, Embedding, Linear, RmsNorm, RotaryEmbedding, RowParallelLinear,
};
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::attention::scaled_dot_product_attention;

// ---------------------------------------------------------------------------
// LlamaConfig
// ---------------------------------------------------------------------------

/// Parsed configuration for a LLaMA model.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
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
    pub tie_word_embeddings: bool,
}

impl LlamaConfig {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> ModelResult<Self> {
        let hidden_size = config
            .hidden_size
            .ok_or_else(|| ModelError::Other("missing hidden_size".into()))?;
        let num_attention_heads = config
            .num_attention_heads
            .ok_or_else(|| ModelError::Other("missing num_attention_heads".into()))?;

        Ok(Self {
            hidden_size,
            num_attention_heads,
            num_kv_heads: config.num_kv_heads().unwrap_or(num_attention_heads),
            num_hidden_layers: config
                .num_hidden_layers
                .ok_or_else(|| ModelError::Other("missing num_hidden_layers".into()))?,
            intermediate_size: config
                .intermediate_size
                .ok_or_else(|| ModelError::Other("missing intermediate_size".into()))?,
            vocab_size: config
                .vocab_size
                .ok_or_else(|| ModelError::Other("missing vocab_size".into()))?,
            max_position_embeddings: config.max_position_embeddings.unwrap_or(4096),
            rms_norm_eps: config.norm_eps(),
            rope_theta: config.rope_theta.unwrap_or(10000.0),
            head_dim: config
                .head_dim()
                .unwrap_or(hidden_size / num_attention_heads),
            tie_word_embeddings: config.tie_word_embeddings.unwrap_or(false),
        })
    }
}

// ---------------------------------------------------------------------------
// LlamaMLP
// ---------------------------------------------------------------------------

/// LLaMA MLP (SiLU-gated feed-forward network).
///
/// Forward: gate_proj(x) → SiLU → * up_proj(x) → down_proj
///
/// Port of: `vllm/model_executor/models/llama.py::LlamaMLP`
pub struct LlamaMLP {
    gate_proj: ColumnParallelLinear,
    up_proj: ColumnParallelLinear,
    down_proj: RowParallelLinear,
}

impl LlamaMLP {
    /// Load MLP weights from a model.
    ///
    /// Weight names: `{prefix}.gate_proj`, `{prefix}.up_proj`, `{prefix}.down_proj`
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

impl Module for LlamaMLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        // SiLU(gate) * up
        let activated = gate.silu()?.mul(&up)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// LlamaAttention
// ---------------------------------------------------------------------------

/// LLaMA multi-head attention with RoPE and optional GQA.
///
/// Forward: q,k,v = separate projections → RoPE → scaled dot-product attention → o_proj
///
/// Port of: `vllm/model_executor/models/llama.py::LlamaAttention`
pub struct LlamaAttention {
    q_proj: ColumnParallelLinear,
    k_proj: ColumnParallelLinear,
    v_proj: ColumnParallelLinear,
    o_proj: RowParallelLinear,
    rotary_emb: RotaryEmbedding,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl LlamaAttention {
    /// Load attention weights.
    ///
    /// Weight names: `{prefix}.q_proj`, `{prefix}.k_proj`, `{prefix}.v_proj`, `{prefix}.o_proj`
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &LlamaConfig,
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
        let head_dim = config.head_dim;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
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
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// Create with zero weights (for testing).
    pub fn zeros(config: &LlamaConfig, dtype: DType, device: &Device) -> ModelResult<Self> {
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
            scale: 1.0 / (config.head_dim as f64).sqrt(),
        })
    }

    /// Forward pass.
    ///
    /// * `hidden_states` — shape `[num_tokens, hidden_size]`
    /// * `positions` — shape `[num_tokens]`
    /// * `kv_cache` — optional `(cached_key, cached_value)` for this layer.
    ///   When `Some`, new K/V are concatenated with the cache and the cache
    ///   is updated in-place. The attention then uses the full KV sequence.
    ///
    /// Returns shape `[num_tokens, hidden_size]`.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut Option<(Tensor, Tensor)>>,
    ) -> ModelResult<Tensor> {
        let num_tokens = hidden_states.dim(0).map_err(ModelError::Candle)?;

        // Q/K/V projections.
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

        // Reshape to [num_tokens, num_heads, head_dim].
        let q = q
            .reshape((num_tokens, self.num_q_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let k = k
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let v = v
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;

        // Apply RoPE.
        let (q, k) = self.rotary_emb.apply(&q, &k, positions)?;

        // Merge with KV cache: concatenate cached K/V with new K/V.
        let (k_full, v_full) = if let Some(cache_slot) = kv_cache {
            if let Some((cached_k, cached_v)) = cache_slot.take() {
                let k_cat = Tensor::cat(&[&cached_k, &k], 0).map_err(ModelError::Candle)?;
                let v_cat = Tensor::cat(&[&cached_v, &v], 0).map_err(ModelError::Candle)?;
                *cache_slot = Some((k_cat.clone(), v_cat.clone()));
                (k_cat, v_cat)
            } else {
                // First call (prefill): populate the cache.
                *cache_slot = Some((k.clone(), v.clone()));
                (k, v)
            }
        } else {
            // No caching requested.
            (k, v)
        };

        // Scaled dot-product attention (handles q_len != kv_len).
        let attn_output = scaled_dot_product_attention(&q, &k_full, &v_full, self.scale)?;

        // Reshape back to [num_tokens, num_q_heads * head_dim].
        let attn_output = attn_output
            .reshape((num_tokens, self.num_q_heads * self.head_dim))
            .map_err(ModelError::Candle)?;

        // Output projection.
        self.o_proj
            .forward(&attn_output)
            .map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// LlamaDecoderLayer
// ---------------------------------------------------------------------------

/// A single LLaMA decoder layer.
///
/// Applies: input_layernorm → attention → residual → post_attention_layernorm → MLP → residual
///
/// Port of: `vllm/model_executor/models/llama.py::LlamaDecoderLayer`
pub struct LlamaDecoderLayer {
    self_attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl LlamaDecoderLayer {
    /// Load a decoder layer.
    ///
    /// Weight names under `{prefix}`:
    /// - `self_attn.{q,k,v,o}_proj`
    /// - `mlp.{gate,up,down}_proj`
    /// - `input_layernorm`
    /// - `post_attention_layernorm`
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &LlamaConfig,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let self_attn = LlamaAttention::load(
            weights,
            &format!("{}.self_attn", prefix),
            config,
            dtype,
            device,
            rank,
            world_size,
        )?;
        let mlp = LlamaMLP::load(weights, &format!("{}.mlp", prefix), dtype, rank, world_size)?;
        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{}.input_layernorm", prefix),
            config.rms_norm_eps,
            dtype,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{}.post_attention_layernorm", prefix),
            config.rms_norm_eps,
            dtype,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    ///
    /// * `hidden_states` — shape `[num_tokens, hidden_size]`
    /// * `positions` — shape `[num_tokens]`
    /// * `kv_cache` — optional KV cache entry for this layer's attention
    ///
    /// Returns hidden_states of same shape.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut Option<(Tensor, Tensor)>>,
    ) -> ModelResult<Tensor> {
        // Pre-attention layernorm + attention + residual.
        let normed = self
            .input_layernorm
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;
        let attn_output = self.self_attn.forward(&normed, positions, kv_cache)?;
        let hidden_states = (hidden_states + attn_output).map_err(ModelError::Candle)?;

        // Post-attention layernorm + MLP + residual.
        let normed = self
            .post_attention_layernorm
            .forward(&hidden_states)
            .map_err(ModelError::Candle)?;
        let mlp_output = self.mlp.forward(&normed).map_err(ModelError::Candle)?;
        let hidden_states = (hidden_states + mlp_output).map_err(ModelError::Candle)?;

        Ok(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// LlamaModel
// ---------------------------------------------------------------------------

/// LLaMA transformer backbone.
///
/// Embedding → N decoder layers → final RMS norm.
///
/// Port of: `vllm/model_executor/models/llama.py::LlamaModel`
pub struct LlamaModel {
    embed_tokens: Embedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RmsNorm,
}

impl LlamaModel {
    /// Load the model backbone.
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &LlamaConfig,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let embed_tokens = Embedding::load(weights, &format!("{}.embed_tokens", prefix), dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = LlamaDecoderLayer::load(
                weights,
                &format!("{}.layers.{}", prefix, i),
                config,
                dtype,
                device,
                rank,
                world_size,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(
            weights,
            &format!("{}.norm", prefix),
            config.rms_norm_eps,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass.
    ///
    /// * `input_ids` — shape `[num_tokens]`
    /// * `positions` — shape `[num_tokens]`
    /// * `kv_cache` — optional per-layer KV cache
    ///
    /// Returns hidden states of shape `[num_tokens, hidden_size]`.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        mut kv_cache: Option<&mut crate::KvCache>,
    ) -> ModelResult<Tensor> {
        let mut hidden_states = self
            .embed_tokens
            .forward(input_ids)
            .map_err(ModelError::Candle)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = kv_cache.as_deref_mut().map(|c| &mut c[i]);
            hidden_states = layer.forward(&hidden_states, positions, layer_cache)?;
        }

        self.norm
            .forward(&hidden_states)
            .map_err(ModelError::Candle)
    }

    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// LlamaForCausalLM
// ---------------------------------------------------------------------------

/// LLaMA for causal language modeling.
///
/// Wraps `LlamaModel` with a language model head that projects hidden states
/// to vocabulary logits.
///
/// Port of: `vllm/model_executor/models/llama.py::LlamaForCausalLM`
pub struct LlamaForCausalLM {
    model: LlamaModel,
    lm_head: Linear,
}

impl LlamaForCausalLM {
    /// Load the full model from weights.
    pub fn load(
        weights: &ModelWeights,
        config: &LlamaConfig,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let model = LlamaModel::load(weights, "model", config, dtype, device, rank, world_size)?;

        let lm_head = if config.tie_word_embeddings {
            // Reuse the embedding weight as the lm_head weight.
            Linear::new(model.embed_tokens.weight().clone(), None)
        } else {
            Linear::load(weights, "lm_head", dtype)?
        };

        Ok(Self { model, lm_head })
    }

    /// Compute logits from hidden states.
    pub fn compute_logits(&self, hidden_states: &Tensor) -> ModelResult<Tensor> {
        self.lm_head
            .forward(hidden_states)
            .map_err(ModelError::Candle)
    }

    /// Access the underlying model backbone.
    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    /// Access the config that was used to build this model.
    pub fn lm_head(&self) -> &Linear {
        &self.lm_head
    }
}

impl crate::Model for LlamaForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCache>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self.model.forward(input_ids, positions, kv_cache)?;
        self.compute_logits(&hidden_states)
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }
}

/// Factory function for the model registry.
pub fn create_llama(
    weights: &ModelWeights,
    config: &HfModelConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let llama_config = LlamaConfig::from_hf_config(config)?;
    let model = LlamaForCausalLM::load(weights, &llama_config, dtype, device, 0, 1)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> LlamaConfig {
        LlamaConfig {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: 8,
            tie_word_embeddings: false,
        }
    }

    fn test_config_gqa() -> LlamaConfig {
        LlamaConfig {
            num_kv_heads: 2, // GQA: 4 q heads, 2 kv heads
            ..test_config()
        }
    }

    #[test]
    fn test_llama_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }"#,
        )
        .unwrap();

        let config = LlamaConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.intermediate_size, 11008);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.head_dim, 128);
        assert!(!config.tie_word_embeddings);
    }

    #[test]
    fn test_llama_mlp_forward_zeros() {
        let config = test_config();
        let mlp = LlamaMLP::zeros(
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
    fn test_llama_attention_forward() {
        let config = test_config();
        let attn = LlamaAttention::zeros(&config, DType::F32, &Device::Cpu).unwrap();

        let num_tokens = 4;
        let x = Tensor::ones(&[num_tokens, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[num_tokens, config.hidden_size]);
    }

    #[test]
    fn test_llama_attention_gqa() {
        let config = test_config_gqa();
        let attn = LlamaAttention::zeros(&config, DType::F32, &Device::Cpu).unwrap();

        let num_tokens = 3;
        let x = Tensor::ones(&[num_tokens, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[num_tokens, config.hidden_size]);
    }

    #[test]
    fn test_llama_attention_single_token() {
        let config = test_config();
        let attn = LlamaAttention::zeros(&config, DType::F32, &Device::Cpu).unwrap();

        let x = Tensor::ones(&[1, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[1, config.hidden_size]);
    }

    #[test]
    fn test_llama_model_from_weights() {
        // Build a tiny LLaMA model from synthetic weights.
        let config = test_config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let device = Device::Cpu;
        let dtype = DType::F32;

        // Generate all the weight tensors needed.
        let mut tensor_specs: Vec<(&str, Vec<usize>)> = Vec::new();

        // Embeddings.
        tensor_specs.push((
            "model.embed_tokens.weight",
            vec![config.vocab_size, config.hidden_size],
        ));

        // Layers.
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let q_size = config.num_attention_heads * config.head_dim;
            let kv_size = config.num_kv_heads * config.head_dim;

            tensor_specs.push((
                Box::leak(format!("{}.self_attn.q_proj.weight", prefix).into_boxed_str()),
                vec![q_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.k_proj.weight", prefix).into_boxed_str()),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.v_proj.weight", prefix).into_boxed_str()),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.o_proj.weight", prefix).into_boxed_str()),
                vec![config.hidden_size, q_size],
            ));

            tensor_specs.push((
                Box::leak(format!("{}.mlp.gate_proj.weight", prefix).into_boxed_str()),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.mlp.up_proj.weight", prefix).into_boxed_str()),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.mlp.down_proj.weight", prefix).into_boxed_str()),
                vec![config.hidden_size, config.intermediate_size],
            ));

            tensor_specs.push((
                Box::leak(format!("{}.input_layernorm.weight", prefix).into_boxed_str()),
                vec![config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.post_attention_layernorm.weight", prefix).into_boxed_str()),
                vec![config.hidden_size],
            ));
        }

        // Final norm.
        tensor_specs.push(("model.norm.weight", vec![config.hidden_size]));

        // LM head.
        tensor_specs.push((
            "lm_head.weight",
            vec![config.vocab_size, config.hidden_size],
        ));

        // Create safetensors file with small random-ish weights.
        create_test_weights(&path, &tensor_specs);

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model = LlamaForCausalLM::load(&weights, &config, dtype, &device, 0, 1).unwrap();

        // Forward pass.
        let input_ids = Tensor::new(&[1u32, 5, 10], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();

        let logits = crate::Model::forward(&model, &input_ids, &positions, None).unwrap();
        assert_eq!(logits.dims(), &[3, config.vocab_size]);
    }

    #[test]
    fn test_llama_model_tied_embeddings() {
        let config = LlamaConfig {
            tie_word_embeddings: true,
            ..test_config()
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let device = Device::Cpu;
        let dtype = DType::F32;

        let mut tensor_specs: Vec<(&str, Vec<usize>)> = Vec::new();
        tensor_specs.push((
            "model.embed_tokens.weight",
            vec![config.vocab_size, config.hidden_size],
        ));

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let q_size = config.num_attention_heads * config.head_dim;
            let kv_size = config.num_kv_heads * config.head_dim;

            tensor_specs.push((
                Box::leak(format!("{}.self_attn.q_proj.weight", prefix).into_boxed_str()),
                vec![q_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.k_proj.weight", prefix).into_boxed_str()),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.v_proj.weight", prefix).into_boxed_str()),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.o_proj.weight", prefix).into_boxed_str()),
                vec![config.hidden_size, q_size],
            ));

            tensor_specs.push((
                Box::leak(format!("{}.mlp.gate_proj.weight", prefix).into_boxed_str()),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.mlp.up_proj.weight", prefix).into_boxed_str()),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.mlp.down_proj.weight", prefix).into_boxed_str()),
                vec![config.hidden_size, config.intermediate_size],
            ));

            tensor_specs.push((
                Box::leak(format!("{}.input_layernorm.weight", prefix).into_boxed_str()),
                vec![config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.post_attention_layernorm.weight", prefix).into_boxed_str()),
                vec![config.hidden_size],
            ));
        }

        tensor_specs.push(("model.norm.weight", vec![config.hidden_size]));
        // No lm_head.weight — tied embeddings!

        create_test_weights(&path, &tensor_specs);

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model = LlamaForCausalLM::load(&weights, &config, dtype, &device, 0, 1).unwrap();

        let input_ids = Tensor::new(&[1u32, 5], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1], &device).unwrap();
        let logits = crate::Model::forward(&model, &input_ids, &positions, None).unwrap();
        assert_eq!(logits.dims(), &[2, config.vocab_size]);
    }

    #[test]
    fn test_llama_registry() {
        let registry = crate::ModelRegistry::default_registry();
        assert!(registry.contains("LlamaForCausalLM"));

        let factory = registry.get("LlamaForCausalLM").unwrap();

        // Create a tiny model via the factory.
        let config = test_config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let device = Device::Cpu;

        let mut tensor_specs: Vec<(&str, Vec<usize>)> = Vec::new();
        tensor_specs.push((
            "model.embed_tokens.weight",
            vec![config.vocab_size, config.hidden_size],
        ));
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let q_size = config.num_attention_heads * config.head_dim;
            let kv_size = config.num_kv_heads * config.head_dim;
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.q_proj.weight", prefix).into_boxed_str()),
                vec![q_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.k_proj.weight", prefix).into_boxed_str()),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.v_proj.weight", prefix).into_boxed_str()),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.o_proj.weight", prefix).into_boxed_str()),
                vec![config.hidden_size, q_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.mlp.gate_proj.weight", prefix).into_boxed_str()),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.mlp.up_proj.weight", prefix).into_boxed_str()),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.mlp.down_proj.weight", prefix).into_boxed_str()),
                vec![config.hidden_size, config.intermediate_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.input_layernorm.weight", prefix).into_boxed_str()),
                vec![config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.post_attention_layernorm.weight", prefix).into_boxed_str()),
                vec![config.hidden_size],
            ));
        }
        tensor_specs.push(("model.norm.weight", vec![config.hidden_size]));
        tensor_specs.push((
            "lm_head.weight",
            vec![config.vocab_size, config.hidden_size],
        ));
        create_test_weights(&path, &tensor_specs);

        let hf_config: HfModelConfig = serde_json::from_str(&format!(
            r#"{{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": {},
                "num_attention_heads": {},
                "num_key_value_heads": {},
                "num_hidden_layers": {},
                "intermediate_size": {},
                "vocab_size": {},
                "max_position_embeddings": {},
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }}"#,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.vocab_size,
            config.max_position_embeddings,
        ))
        .unwrap();

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model = factory(&weights, &hf_config, DType::F32, &device).unwrap();

        let input_ids = Tensor::new(&[1u32, 2, 3], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();
        let logits = model.forward(&input_ids, &positions, None).unwrap();
        assert_eq!(logits.dims(), &[3, config.vocab_size]);
    }

    #[test]
    fn test_llama_kv_cache_prefill_and_decode() {
        // Verify that KV cache is populated on prefill and used on decode,
        // and that the decode output shape is correct.
        let config = test_config();
        let attn = LlamaAttention::zeros(&config, DType::F32, &Device::Cpu).unwrap();

        // Prefill: 4 tokens.
        let x = Tensor::ones(&[4, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();
        let mut cache: Option<(Tensor, Tensor)> = None;

        let out = attn.forward(&x, &positions, Some(&mut cache)).unwrap();
        assert_eq!(out.dims(), &[4, config.hidden_size]);

        // Cache should now hold K/V of length 4.
        let (cached_k, cached_v) = cache.as_ref().unwrap();
        assert_eq!(cached_k.dim(0).unwrap(), 4);
        assert_eq!(cached_v.dim(0).unwrap(), 4);

        // Decode: 1 token at position 4.
        let x_decode = Tensor::ones(&[1, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let pos_decode = Tensor::new(&[4u32], &Device::Cpu).unwrap();

        let out_decode = attn
            .forward(&x_decode, &pos_decode, Some(&mut cache))
            .unwrap();
        assert_eq!(out_decode.dims(), &[1, config.hidden_size]);

        // Cache should now hold K/V of length 5.
        let (cached_k, cached_v) = cache.as_ref().unwrap();
        assert_eq!(cached_k.dim(0).unwrap(), 5);
        assert_eq!(cached_v.dim(0).unwrap(), 5);
    }

    #[test]
    fn test_llama_model_kv_cache_e2e() {
        // End-to-end: run a tiny model with KV cache through prefill + decode.
        let config = test_config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let device = Device::Cpu;
        let dtype = DType::F32;

        let mut tensor_specs: Vec<(&str, Vec<usize>)> = Vec::new();
        tensor_specs.push((
            "model.embed_tokens.weight",
            vec![config.vocab_size, config.hidden_size],
        ));
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let q_size = config.num_attention_heads * config.head_dim;
            let kv_size = config.num_kv_heads * config.head_dim;
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.q_proj.weight", prefix).into_boxed_str()),
                vec![q_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.k_proj.weight", prefix).into_boxed_str()),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.v_proj.weight", prefix).into_boxed_str()),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.self_attn.o_proj.weight", prefix).into_boxed_str()),
                vec![config.hidden_size, q_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.mlp.gate_proj.weight", prefix).into_boxed_str()),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.mlp.up_proj.weight", prefix).into_boxed_str()),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.mlp.down_proj.weight", prefix).into_boxed_str()),
                vec![config.hidden_size, config.intermediate_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.input_layernorm.weight", prefix).into_boxed_str()),
                vec![config.hidden_size],
            ));
            tensor_specs.push((
                Box::leak(format!("{}.post_attention_layernorm.weight", prefix).into_boxed_str()),
                vec![config.hidden_size],
            ));
        }
        tensor_specs.push(("model.norm.weight", vec![config.hidden_size]));
        tensor_specs.push((
            "lm_head.weight",
            vec![config.vocab_size, config.hidden_size],
        ));
        create_test_weights(&path, &tensor_specs);

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model = LlamaForCausalLM::load(&weights, &config, dtype, &device, 0, 1).unwrap();

        // Prefill with 3 tokens and KV cache.
        let mut kv_cache: crate::KvCache = vec![None; model.model().num_layers()];
        let input_ids = Tensor::new(&[1u32, 5, 10], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();
        let logits =
            crate::Model::forward(&model, &input_ids, &positions, Some(&mut kv_cache)).unwrap();
        assert_eq!(logits.dims(), &[3, config.vocab_size]);

        // All layers should have caches populated.
        for (i, entry) in kv_cache.iter().enumerate() {
            let (k, v) = entry
                .as_ref()
                .unwrap_or_else(|| panic!("layer {i} cache empty"));
            assert_eq!(k.dim(0).unwrap(), 3, "layer {i} K length");
            assert_eq!(v.dim(0).unwrap(), 3, "layer {i} V length");
        }

        // Decode: 1 token at position 3.
        let decode_ids = Tensor::new(&[15u32], &device).unwrap();
        let decode_pos = Tensor::new(&[3u32], &device).unwrap();
        let logits2 =
            crate::Model::forward(&model, &decode_ids, &decode_pos, Some(&mut kv_cache)).unwrap();
        assert_eq!(logits2.dims(), &[1, config.vocab_size]);

        // Caches should now have length 4.
        for (i, entry) in kv_cache.iter().enumerate() {
            let (k, v) = entry.as_ref().unwrap();
            assert_eq!(k.dim(0).unwrap(), 4, "layer {i} K length after decode");
            assert_eq!(v.dim(0).unwrap(), 4, "layer {i} V length after decode");
        }
    }

    // -----------------------------------------------------------------------
    // Test helper: create a safetensors file with small constant weights.
    // -----------------------------------------------------------------------

    fn create_test_weights(path: &std::path::Path, specs: &[(&str, Vec<usize>)]) {
        use safetensors::tensor::TensorView;

        // Generate weight data (small values to avoid numerical issues).
        let mut all_data: Vec<Vec<u8>> = Vec::new();
        for (_, shape) in specs {
            let num_elements: usize = shape.iter().product();
            // Use small deterministic values: 0.01 for all elements.
            // Norm weights use 1.0 so that RmsNorm doesn't distort values.
            let data: Vec<u8> = (0..num_elements)
                .flat_map(|_| 0.01f32.to_le_bytes())
                .collect();
            all_data.push(data);
        }

        // Override norm weights with 1.0.
        for (i, (name, shape)) in specs.iter().enumerate() {
            if name.contains("layernorm") || (*name == "model.norm.weight") {
                let num_elements: usize = shape.iter().product();
                all_data[i] = (0..num_elements)
                    .flat_map(|_| 1.0f32.to_le_bytes())
                    .collect();
            }
        }

        let views: Vec<(&str, TensorView<'_>)> = specs
            .iter()
            .zip(all_data.iter())
            .map(|((name, shape), data)| {
                (
                    *name,
                    TensorView::new(safetensors::Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        let bytes = safetensors::tensor::serialize(views, None).unwrap();
        std::fs::write(path, bytes).unwrap();
    }
}
