// SPDX-License-Identifier: Apache-2.0
//! BitsAndBytes quantized LLaMA/Qwen2 model architecture (NF4 + INT8).
//!
//! Mirrors `llama.rs` but uses `BnbLinear` for all linear projections.
//! Embedding, norms, and lm_head stay in float (BnB typically doesn't
//! quantize these). KV cache uses the working dtype (usually bf16).

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::bnb_config::BnbQuantizeConfig;
use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{BnbLayerConfig, BnbLinear, Embedding, Linear, RmsNorm, RotaryEmbedding};
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::attention::attention_with_cache;
use crate::llama::LlamaConfig;
use crate::qwen2::Qwen2Config;

// ---------------------------------------------------------------------------
// BnbLlamaMLP
// ---------------------------------------------------------------------------

struct BnbLlamaMLP {
    gate_proj: BnbLinear,
    up_proj: BnbLinear,
    down_proj: BnbLinear,
}

impl BnbLlamaMLP {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &LlamaConfig,
        bnb: &BnbLayerConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        Ok(Self {
            gate_proj: BnbLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                bnb,
                config.intermediate_size,
                config.hidden_size,
                dtype,
                device,
            )?,
            up_proj: BnbLinear::from_weights(
                weights,
                &format!("{prefix}.up_proj"),
                bnb,
                config.intermediate_size,
                config.hidden_size,
                dtype,
                device,
            )?,
            down_proj: BnbLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                bnb,
                config.hidden_size,
                config.intermediate_size,
                dtype,
                device,
            )?,
        })
    }
}

impl Module for BnbLlamaMLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = crate::ops::silu_and_mul(&gate, &up)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// BnbLlamaAttention
// ---------------------------------------------------------------------------

struct BnbLlamaAttention {
    q_proj: BnbLinear,
    k_proj: BnbLinear,
    v_proj: BnbLinear,
    o_proj: BnbLinear,
    rotary_emb: RotaryEmbedding,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    sliding_window: Option<usize>,
}

impl BnbLlamaAttention {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &LlamaConfig,
        bnb: &BnbLayerConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let q_proj = BnbLinear::from_weights(
            weights,
            &format!("{prefix}.q_proj"),
            bnb,
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            dtype,
            device,
        )?;
        let k_proj = BnbLinear::from_weights(
            weights,
            &format!("{prefix}.k_proj"),
            bnb,
            config.num_kv_heads * config.head_dim,
            config.hidden_size,
            dtype,
            device,
        )?;
        let v_proj = BnbLinear::from_weights(
            weights,
            &format!("{prefix}.v_proj"),
            bnb,
            config.num_kv_heads * config.head_dim,
            config.hidden_size,
            dtype,
            device,
        )?;
        let o_proj = BnbLinear::from_weights(
            weights,
            &format!("{prefix}.o_proj"),
            bnb,
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            dtype,
            device,
        )?;

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
// BnbLlamaDecoderLayer
// ---------------------------------------------------------------------------

struct BnbLlamaDecoderLayer {
    self_attn: BnbLlamaAttention,
    mlp: BnbLlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl BnbLlamaDecoderLayer {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &LlamaConfig,
        bnb: &BnbLayerConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let self_attn = BnbLlamaAttention::load(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            bnb,
            dtype,
            device,
        )?;
        let mlp = BnbLlamaMLP::load(
            weights,
            &format!("{prefix}.mlp"),
            config,
            bnb,
            dtype,
            device,
        )?;
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
        let hidden_states = (hidden_states + attn_output).map_err(ModelError::Candle)?;

        let normed = crate::ops::rms_norm(&hidden_states, &self.post_attention_layernorm)
            .map_err(ModelError::Candle)?;
        let mlp_output = self.mlp.forward(&normed).map_err(ModelError::Candle)?;
        let hidden_states = (hidden_states + mlp_output).map_err(ModelError::Candle)?;

        Ok(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// BnbLlamaModel
// ---------------------------------------------------------------------------

struct BnbLlamaModel {
    embed_tokens: Embedding,
    layers: Vec<BnbLlamaDecoderLayer>,
    norm: RmsNorm,
}

impl BnbLlamaModel {
    fn load(
        weights: &ModelWeights,
        config: &LlamaConfig,
        bnb: &BnbLayerConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens", dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = BnbLlamaDecoderLayer::load(
                weights,
                &format!("model.layers.{i}"),
                config,
                bnb,
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
// BnbLlamaForCausalLM
// ---------------------------------------------------------------------------

/// BitsAndBytes quantized LLaMA for causal language modeling (NF4 + INT8).
pub struct BnbLlamaForCausalLM {
    model: BnbLlamaModel,
    lm_head: Linear,
}

impl BnbLlamaForCausalLM {
    pub fn load(
        weights: &ModelWeights,
        config: &LlamaConfig,
        bnb: &BnbLayerConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let model = BnbLlamaModel::load(weights, config, bnb, dtype, device)?;

        // lm_head: usually float, not quantized.
        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight().clone(), None)
        } else if weights.contains("lm_head.weight.absmax") || weights.contains("lm_head.SCB") {
            // Rare: some BnB models quantize lm_head too (NF4 or INT8).
            let bnb_lm = BnbLinear::from_weights(
                weights,
                "lm_head",
                bnb,
                config.vocab_size,
                config.hidden_size,
                dtype,
                device,
            )?;
            let w = bnb_lm.dequantize()?;
            Linear::new(w, None)
        } else {
            Linear::load(weights, "lm_head", dtype)?
        };

        Ok(Self { model, lm_head })
    }
}

impl crate::Model for BnbLlamaForCausalLM {
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
// Factory functions for the BnB registry
// ---------------------------------------------------------------------------

/// BnB factory function signature.
pub type BnbModelFactory = fn(
    weights: &ModelWeights,
    config: &HfModelConfig,
    bnb_config: &BnbQuantizeConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>>;

/// Create a BnB LLaMA model (NF4 or INT8).
pub fn create_llama_bnb(
    weights: &ModelWeights,
    config: &HfModelConfig,
    bnb_config: &BnbQuantizeConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let llama_config = LlamaConfig::from_hf_config(config)?;
    let bnb = bnb_config.to_bnb_layer_config();
    let model = BnbLlamaForCausalLM::load(weights, &llama_config, &bnb, dtype, device)?;
    Ok(Box::new(model))
}

/// Create a BnB Qwen2 model (same arch with Qwen2-specific config defaults).
pub fn create_qwen2_bnb(
    weights: &ModelWeights,
    config: &HfModelConfig,
    bnb_config: &BnbQuantizeConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let qwen2_config = Qwen2Config::from_hf_config(config)?;
    let bnb = bnb_config.to_bnb_layer_config();
    let model = BnbLlamaForCausalLM::load(weights, &qwen2_config.0, &bnb, dtype, device)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bnb_factory_types() {
        // Just verify the factory function signatures compile.
        let _: BnbModelFactory = create_llama_bnb;
        let _: BnbModelFactory = create_qwen2_bnb;
    }
}
