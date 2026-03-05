// SPDX-License-Identifier: Apache-2.0
//! GPTQ-quantized LLaMA/Qwen2 model architecture.
//!
//! Mirrors `llama.rs` but uses `GptqLinear` for all linear projections.
//! Embedding, norms, and lm_head stay in float (GPTQ typically doesn't
//! quantize these). KV cache uses the scales dtype (usually f16).

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::gptq_config::GptqQuantizeConfig;
use vllm_model::layers::{Embedding, GptqConfig, GptqLinear, Linear, RmsNorm, RotaryEmbedding};
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::attention::attention_with_cache;
use crate::llama::LlamaConfig;
use crate::qwen2::Qwen2Config;

// ---------------------------------------------------------------------------
// GptqLlamaMLP
// ---------------------------------------------------------------------------

struct GptqLlamaMLP {
    gate_proj: GptqLinear,
    up_proj: GptqLinear,
    down_proj: GptqLinear,
}

impl GptqLlamaMLP {
    fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        gptq: &GptqConfig,
        device: &Device,
    ) -> ModelResult<Self> {
        Ok(Self {
            gate_proj: GptqLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                gptq,
                device,
            )?,
            up_proj: GptqLinear::from_weights(weights, &format!("{prefix}.up_proj"), gptq, device)?,
            down_proj: GptqLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                gptq,
                device,
            )?,
        })
    }
}

impl Module for GptqLlamaMLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = crate::ops::silu_and_mul(&gate, &up)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// GptqLlamaAttention
// ---------------------------------------------------------------------------

struct GptqLlamaAttention {
    q_proj: GptqLinear,
    k_proj: GptqLinear,
    v_proj: GptqLinear,
    o_proj: GptqLinear,
    rotary_emb: RotaryEmbedding,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    sliding_window: Option<usize>,
}

impl GptqLlamaAttention {
    fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &LlamaConfig,
        gptq: &GptqConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let q_proj = GptqLinear::from_weights(weights, &format!("{prefix}.q_proj"), gptq, device)?;
        let k_proj = GptqLinear::from_weights(weights, &format!("{prefix}.k_proj"), gptq, device)?;
        let v_proj = GptqLinear::from_weights(weights, &format!("{prefix}.v_proj"), gptq, device)?;
        let o_proj = GptqLinear::from_weights(weights, &format!("{prefix}.o_proj"), gptq, device)?;

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
// GptqLlamaDecoderLayer
// ---------------------------------------------------------------------------

struct GptqLlamaDecoderLayer {
    self_attn: GptqLlamaAttention,
    mlp: GptqLlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl GptqLlamaDecoderLayer {
    fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &LlamaConfig,
        gptq: &GptqConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let self_attn = GptqLlamaAttention::load(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            gptq,
            dtype,
            device,
        )?;
        let mlp = GptqLlamaMLP::load(weights, &format!("{prefix}.mlp"), gptq, device)?;
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
        let normed = crate::ops::rms_norm(hidden_states, &self.input_layernorm)
            .map_err(ModelError::Candle)?;
        let attn_output = self.self_attn.forward(&normed, positions, kv_cache)?;

        // Fused residual add + post-attention layernorm.
        let (normed, hidden_states) = crate::ops::fused_add_rms_norm(
            &attn_output,
            hidden_states,
            &self.post_attention_layernorm,
        )
        .map_err(ModelError::Candle)?;

        let mlp_output = self.mlp.forward(&normed).map_err(ModelError::Candle)?;
        let hidden_states = (hidden_states + mlp_output).map_err(ModelError::Candle)?;

        Ok(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// GptqLlamaModel
// ---------------------------------------------------------------------------

struct GptqLlamaModel {
    embed_tokens: Embedding,
    layers: Vec<GptqLlamaDecoderLayer>,
    norm: RmsNorm,
}

impl GptqLlamaModel {
    fn load(
        weights: &mut ModelWeights,
        config: &LlamaConfig,
        gptq: &GptqConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens", dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = GptqLlamaDecoderLayer::load(
                weights,
                &format!("model.layers.{i}"),
                config,
                gptq,
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

        crate::ops::rms_norm(&hidden_states, &self.norm).map_err(ModelError::Candle)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// GptqLlamaForCausalLM
// ---------------------------------------------------------------------------

/// GPTQ-quantized LLaMA for causal language modeling.
pub struct GptqLlamaForCausalLM {
    model: GptqLlamaModel,
    lm_head: Linear,
}

impl GptqLlamaForCausalLM {
    pub fn load(
        weights: &mut ModelWeights,
        config: &LlamaConfig,
        gptq: &GptqConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let model = GptqLlamaModel::load(weights, config, gptq, dtype, device)?;

        // lm_head: usually float, not quantized.
        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight().clone(), None)
        } else if weights.contains("lm_head.qweight") {
            // Rare: some GPTQ models quantize lm_head too.
            let gptq_lm = GptqLinear::from_weights(weights, "lm_head", gptq, device)?;
            // Dequantize to a regular Linear for simplicity.
            let w = gptq_lm.dequantize()?;
            Linear::new(w, None)
        } else {
            Linear::load(weights, "lm_head", dtype)?
        };

        Ok(Self { model, lm_head })
    }
}

impl crate::Model for GptqLlamaForCausalLM {
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
// Factory functions for the GPTQ registry
// ---------------------------------------------------------------------------

/// GPTQ factory function signature.
pub type GptqModelFactory = fn(
    weights: &mut ModelWeights,
    config: &HfModelConfig,
    gptq_config: &GptqQuantizeConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>>;

/// Create a GPTQ LLaMA model.
pub fn create_llama_gptq(
    weights: &mut ModelWeights,
    config: &HfModelConfig,
    gptq_config: &GptqQuantizeConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let llama_config = LlamaConfig::from_hf_config(config)?;
    let gptq = gptq_config.to_gptq_config();
    let model = GptqLlamaForCausalLM::load(weights, &llama_config, &gptq, dtype, device)?;
    Ok(Box::new(model))
}

/// Create a GPTQ Qwen2 model (same arch with Qwen2-specific config defaults).
pub fn create_qwen2_gptq(
    weights: &mut ModelWeights,
    config: &HfModelConfig,
    gptq_config: &GptqQuantizeConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let qwen2_config = Qwen2Config::from_hf_config(config)?;
    let gptq = gptq_config.to_gptq_config();
    let model = GptqLlamaForCausalLM::load(weights, &qwen2_config.0, &gptq, dtype, device)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gptq_factory_types() {
        // Just verify the factory function signatures compile.
        let _: GptqModelFactory = create_llama_gptq;
        let _: GptqModelFactory = create_qwen2_gptq;
    }
}
