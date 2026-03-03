// SPDX-License-Identifier: Apache-2.0
//! Quantized Granite model architecture using GGUF weights.
//!
//! Reuses quantized LLaMA components with Granite's 4 scalar multipliers:
//! - `embedding_multiplier` — scales embeddings after lookup
//! - `residual_multiplier` — scales attention/MLP outputs before residual add
//! - `attention_multiplier` — replaces the standard 1/sqrt(head_dim) scaling
//! - `logits_scaling` — divides logits before softmax

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::gguf::GgufFile;
use vllm_model::layers::{Embedding, QuantizedLinear, RmsNorm};
use vllm_model::weight::HfModelConfig;

use crate::granite::GraniteConfig;
use crate::quantized_llama::{QuantizedLlamaAttention, QuantizedLlamaMLP, precompute_freqs_cis};

// ---------------------------------------------------------------------------
// QuantizedGraniteDecoderLayer
// ---------------------------------------------------------------------------

/// A single quantized Granite decoder layer.
struct QuantizedGraniteDecoderLayer {
    self_attn: QuantizedLlamaAttention,
    mlp: QuantizedLlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    residual_multiplier: f64,
}

impl QuantizedGraniteDecoderLayer {
    fn load(
        gguf: &mut GgufFile,
        prefix: &str,
        config: &GraniteConfig,
        cos: Tensor,
        sin: Tensor,
        device: &Device,
    ) -> ModelResult<Self> {
        let mut self_attn =
            QuantizedLlamaAttention::load(gguf, prefix, &config.llama, cos, sin, device)?;
        // Override attention scaling with Granite's multiplier.
        self_attn.scale = config.attention_multiplier;

        let mlp = QuantizedLlamaMLP::load(gguf, prefix, device)?;

        let input_ln_weight = gguf
            .tensor(&format!("{prefix}.attn_norm.weight"), device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize norm: {e}")))?;
        let input_layernorm = RmsNorm::new(input_ln_weight, config.llama.rms_norm_eps);

        let post_ln_weight = gguf
            .tensor(&format!("{prefix}.ffn_norm.weight"), device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize norm: {e}")))?;
        let post_attention_layernorm = RmsNorm::new(post_ln_weight, config.llama.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_multiplier: config.residual_multiplier,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<Tensor> {
        // Pre-attention norm.
        let normed = crate::ops::rms_norm(hidden_states, &self.input_layernorm)
            .map_err(ModelError::Candle)?;

        // Attention.
        let attn_output = self.self_attn.forward(&normed, positions, kv_cache)?;

        // Residual with scaled attention output.
        let hidden_states = (hidden_states
            + attn_output
                .affine(self.residual_multiplier, 0.0)
                .map_err(ModelError::Candle)?)
        .map_err(ModelError::Candle)?;

        // Post-attention norm.
        let normed = crate::ops::rms_norm(&hidden_states, &self.post_attention_layernorm)
            .map_err(ModelError::Candle)?;

        // MLP with scaled output + residual.
        let mlp_output = self.mlp.forward(&normed).map_err(ModelError::Candle)?;
        let hidden_states = (hidden_states
            + mlp_output
                .affine(self.residual_multiplier, 0.0)
                .map_err(ModelError::Candle)?)
        .map_err(ModelError::Candle)?;

        Ok(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// QuantizedGraniteModel
// ---------------------------------------------------------------------------

/// Quantized Granite transformer backbone.
struct QuantizedGraniteModel {
    embed_tokens: Embedding,
    layers: Vec<QuantizedGraniteDecoderLayer>,
    norm: RmsNorm,
    embedding_multiplier: f64,
}

impl QuantizedGraniteModel {
    fn load(gguf: &mut GgufFile, config: &GraniteConfig, device: &Device) -> ModelResult<Self> {
        let embed_weight = gguf
            .tensor("token_embd.weight", device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize embedding: {e}")))?;
        let embed_tokens = Embedding::new(embed_weight);

        let (cos, sin) = precompute_freqs_cis(
            config.llama.head_dim,
            config.llama.max_position_embeddings,
            config.llama.rope_theta,
            device,
        )?;

        let mut layers = Vec::with_capacity(config.llama.num_hidden_layers);
        for i in 0..config.llama.num_hidden_layers {
            let layer = QuantizedGraniteDecoderLayer::load(
                gguf,
                &format!("blk.{i}"),
                config,
                cos.clone(),
                sin.clone(),
                device,
            )?;
            layers.push(layer);
        }

        let norm_weight = gguf
            .tensor("output_norm.weight", device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize norm: {e}")))?;
        let norm = RmsNorm::new(norm_weight, config.llama.rms_norm_eps);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            embedding_multiplier: config.embedding_multiplier,
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

        // Scale embeddings.
        hidden_states = hidden_states
            .affine(self.embedding_multiplier, 0.0)
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
// QuantizedGraniteForCausalLM
// ---------------------------------------------------------------------------

/// Quantized Granite for causal language modeling.
pub struct QuantizedGraniteForCausalLM {
    model: QuantizedGraniteModel,
    lm_head: QuantizedLinear,
    logits_scaling: f64,
}

impl QuantizedGraniteForCausalLM {
    /// Load the full quantized model from a GGUF file.
    pub fn load(gguf: &mut GgufFile, config: &GraniteConfig, device: &Device) -> ModelResult<Self> {
        let model = QuantizedGraniteModel::load(gguf, config, device)?;

        let has_output = gguf.tensor_names().contains(&"output.weight");
        let lm_head = if has_output {
            QuantizedLinear::from_gguf(gguf, "output.weight", device)?
        } else {
            let qt = gguf.tensor("token_embd.weight", device)?;
            QuantizedLinear::from_qtensor(qt)?
        };

        Ok(Self {
            model,
            lm_head,
            logits_scaling: config.logits_scaling,
        })
    }
}

impl crate::Model for QuantizedGraniteForCausalLM {
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
        let logits = logits
            .affine(1.0 / self.logits_scaling, 0.0)
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
// Factory
// ---------------------------------------------------------------------------

/// Create a quantized Granite model from a GGUF file.
pub fn create_granite_gguf(
    gguf: &mut GgufFile,
    config: &HfModelConfig,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let granite_config = GraniteConfig::from_hf_config(config)?;
    let model = QuantizedGraniteForCausalLM::load(gguf, &granite_config, device)?;
    Ok(Box::new(model))
}
