// SPDX-License-Identifier: Apache-2.0
//! AWQ-quantized LLaMA/Qwen2 model architecture.
//!
//! Mirrors `gptq_llama.rs` but uses `AwqLinear` for all linear projections.
//! Embedding, norms, and lm_head stay in float (AWQ typically doesn't
//! quantize these). KV cache uses the scales dtype (usually f16).

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::awq_config::AwqQuantizeConfig;
use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{AwqConfig, AwqLinear, Embedding, Linear, RmsNorm, RotaryEmbedding};
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::attention::attention_with_cache;
use crate::llama::LlamaConfig;
use crate::qwen2::Qwen2Config;

// ---------------------------------------------------------------------------
// AwqLlamaMLP
// ---------------------------------------------------------------------------

struct AwqLlamaMLP {
    gate_proj: AwqLinear,
    up_proj: AwqLinear,
    down_proj: AwqLinear,
}

impl AwqLlamaMLP {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        awq: &AwqConfig,
        device: &Device,
    ) -> ModelResult<Self> {
        Ok(Self {
            gate_proj: AwqLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                awq,
                device,
            )?,
            up_proj: AwqLinear::from_weights(weights, &format!("{prefix}.up_proj"), awq, device)?,
            down_proj: AwqLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                awq,
                device,
            )?,
        })
    }
}

impl Module for AwqLlamaMLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = gate.silu()?.mul(&up)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// AwqLlamaAttention
// ---------------------------------------------------------------------------

struct AwqLlamaAttention {
    q_proj: AwqLinear,
    k_proj: AwqLinear,
    v_proj: AwqLinear,
    o_proj: AwqLinear,
    rotary_emb: RotaryEmbedding,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    sliding_window: Option<usize>,
}

impl AwqLlamaAttention {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &LlamaConfig,
        awq: &AwqConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let q_proj = AwqLinear::from_weights(weights, &format!("{prefix}.q_proj"), awq, device)?;
        let k_proj = AwqLinear::from_weights(weights, &format!("{prefix}.k_proj"), awq, device)?;
        let v_proj = AwqLinear::from_weights(weights, &format!("{prefix}.v_proj"), awq, device)?;
        let o_proj = AwqLinear::from_weights(weights, &format!("{prefix}.o_proj"), awq, device)?;

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
            sliding_window: config.sliding_window,
        })
    }

    fn forward(
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
// AwqLlamaDecoderLayer
// ---------------------------------------------------------------------------

struct AwqLlamaDecoderLayer {
    self_attn: AwqLlamaAttention,
    mlp: AwqLlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl AwqLlamaDecoderLayer {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &LlamaConfig,
        awq: &AwqConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let self_attn = AwqLlamaAttention::load(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            awq,
            dtype,
            device,
        )?;
        let mlp = AwqLlamaMLP::load(weights, &format!("{prefix}.mlp"), awq, device)?;
        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
            dtype,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
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

    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<Tensor> {
        let normed = self
            .input_layernorm
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;
        let attn_output = self.self_attn.forward(&normed, positions, kv_cache)?;
        let hidden_states = (hidden_states + attn_output).map_err(ModelError::Candle)?;

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
// AwqLlamaModel
// ---------------------------------------------------------------------------

struct AwqLlamaModel {
    embed_tokens: Embedding,
    layers: Vec<AwqLlamaDecoderLayer>,
    norm: RmsNorm,
}

impl AwqLlamaModel {
    fn load(
        weights: &ModelWeights,
        config: &LlamaConfig,
        awq: &AwqConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens", dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = AwqLlamaDecoderLayer::load(
                weights,
                &format!("model.layers.{i}"),
                config,
                awq,
                dtype,
                device,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps, dtype)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        mut kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let mut hidden_states = self
            .embed_tokens
            .forward(input_ids)
            .map_err(ModelError::Candle)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_handle = kv_cache.as_mut().map(|s| s.layer_handle(i));
            hidden_states = layer.forward(&hidden_states, positions, layer_handle)?;
        }

        self.norm
            .forward(&hidden_states)
            .map_err(ModelError::Candle)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// AwqLlamaForCausalLM
// ---------------------------------------------------------------------------

/// AWQ-quantized LLaMA for causal language modeling.
pub struct AwqLlamaForCausalLM {
    model: AwqLlamaModel,
    lm_head: Linear,
}

impl AwqLlamaForCausalLM {
    pub fn load(
        weights: &ModelWeights,
        config: &LlamaConfig,
        awq: &AwqConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let model = AwqLlamaModel::load(weights, config, awq, dtype, device)?;

        // lm_head: usually float, not quantized.
        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight().clone(), None)
        } else if weights.contains("lm_head.qweight") {
            // Rare: some AWQ models quantize lm_head too.
            let awq_lm = AwqLinear::from_weights(weights, "lm_head", awq, device)?;
            let w = awq_lm.dequantize()?;
            Linear::new(w, None)
        } else {
            Linear::load(weights, "lm_head", dtype)?
        };

        Ok(Self { model, lm_head })
    }
}

impl crate::Model for AwqLlamaForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self.model.forward(input_ids, positions, kv_cache)?;
        let logits = self
            .lm_head
            .forward(&hidden_states)
            .map_err(ModelError::Candle)?;
        logits.to_dtype(DType::F32).map_err(ModelError::Candle)
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn hidden_states(&self, input_ids: &Tensor, positions: &Tensor) -> ModelResult<Tensor> {
        self.model.forward(input_ids, positions, None)
    }
}

// ---------------------------------------------------------------------------
// Factory functions for the AWQ registry
// ---------------------------------------------------------------------------

/// AWQ factory function signature.
pub type AwqModelFactory = fn(
    weights: &ModelWeights,
    config: &HfModelConfig,
    awq_config: &AwqQuantizeConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>>;

/// Create an AWQ LLaMA model.
pub fn create_llama_awq(
    weights: &ModelWeights,
    config: &HfModelConfig,
    awq_config: &AwqQuantizeConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let llama_config = LlamaConfig::from_hf_config(config)?;
    let awq = awq_config.to_awq_config();
    let model = AwqLlamaForCausalLM::load(weights, &llama_config, &awq, dtype, device)?;
    Ok(Box::new(model))
}

/// Create an AWQ Qwen2 model (same arch with Qwen2-specific config defaults).
pub fn create_qwen2_awq(
    weights: &ModelWeights,
    config: &HfModelConfig,
    awq_config: &AwqQuantizeConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let qwen2_config = Qwen2Config::from_hf_config(config)?;
    let awq = awq_config.to_awq_config();
    let model = AwqLlamaForCausalLM::load(weights, &qwen2_config.0, &awq, dtype, device)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_awq_factory_types() {
        // Just verify the factory function signatures compile.
        let _: AwqModelFactory = create_llama_awq;
        let _: AwqModelFactory = create_qwen2_awq;
    }
}
