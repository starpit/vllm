// SPDX-License-Identifier: Apache-2.0
//! Quantized Qwen2 model architecture using GGUF weights.
//!
//! Qwen2 is architecturally identical to LLaMA with one key difference:
//! Q/K/V projections include bias terms. In GGUF, these are stored as
//! separate tensors (`blk.N.attn_q.bias`, etc.) that must be dequantized
//! and added after the quantized matmul.
//!
//! Reuses `QuantizedLlamaMLP` from `quantized_llama.rs` (same activation,
//! same tensor names).

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::gguf::GgufFile;
use vllm_model::layers::{Embedding, QuantizedLinear, RmsNorm};
use vllm_model::weight::HfModelConfig;

use crate::attention::attention_with_cache;
use crate::llama::LlamaConfig;
use crate::quantized_llama::{QuantizedLlamaMLP, apply_interleaved_rope, precompute_freqs_cis};
use crate::qwen2::Qwen2Config;

// ---------------------------------------------------------------------------
// Helper: load a dequantized bias vector from GGUF (if present)
// ---------------------------------------------------------------------------

fn load_bias(gguf: &mut GgufFile, name: &str, device: &Device) -> ModelResult<Option<Tensor>> {
    if !gguf.tensor_names().contains(&name) {
        return Ok(None);
    }
    let bias = gguf
        .tensor(name, device)?
        .dequantize(device)
        .map_err(|e| ModelError::Other(format!("dequantize bias: {e}")))?;
    Ok(Some(bias))
}

// ---------------------------------------------------------------------------
// QuantizedQwen2Attention
// ---------------------------------------------------------------------------

/// Qwen2 attention with quantized Q/K/V/O projections and optional QKV bias.
struct QuantizedQwen2Attention {
    q_proj: QuantizedLinear,
    k_proj: QuantizedLinear,
    v_proj: QuantizedLinear,
    o_proj: QuantizedLinear,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    cos: Tensor,
    sin: Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    sliding_window: Option<usize>,
}

impl QuantizedQwen2Attention {
    fn load(
        gguf: &mut GgufFile,
        prefix: &str,
        config: &LlamaConfig,
        cos: Tensor,
        sin: Tensor,
        device: &Device,
    ) -> ModelResult<Self> {
        let q_proj = QuantizedLinear::from_gguf(gguf, &format!("{prefix}.attn_q.weight"), device)?;
        let k_proj = QuantizedLinear::from_gguf(gguf, &format!("{prefix}.attn_k.weight"), device)?;
        let v_proj = QuantizedLinear::from_gguf(gguf, &format!("{prefix}.attn_v.weight"), device)?;
        let o_proj =
            QuantizedLinear::from_gguf(gguf, &format!("{prefix}.attn_output.weight"), device)?;

        let q_bias = load_bias(gguf, &format!("{prefix}.attn_q.bias"), device)?;
        let k_bias = load_bias(gguf, &format!("{prefix}.attn_k.bias"), device)?;
        let v_bias = load_bias(gguf, &format!("{prefix}.attn_v.bias"), device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_bias,
            k_bias,
            v_bias,
            cos,
            sin,
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

        let mut q = self
            .q_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;
        let mut k = self
            .k_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;
        let mut v = self
            .v_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;

        // Add bias if present.
        if let Some(ref bias) = self.q_bias {
            q = q.broadcast_add(bias).map_err(ModelError::Candle)?;
        }
        if let Some(ref bias) = self.k_bias {
            k = k.broadcast_add(bias).map_err(ModelError::Candle)?;
        }
        if let Some(ref bias) = self.v_bias {
            v = v.broadcast_add(bias).map_err(ModelError::Candle)?;
        }

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

        // Apply interleaved RoPE (GGML convention).
        let q = apply_interleaved_rope(&q, &self.cos, &self.sin, positions)?;
        let k = apply_interleaved_rope(&k, &self.cos, &self.sin, positions)?;

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
// QuantizedQwen2DecoderLayer
// ---------------------------------------------------------------------------

struct QuantizedQwen2DecoderLayer {
    self_attn: QuantizedQwen2Attention,
    mlp: QuantizedLlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedQwen2DecoderLayer {
    fn load(
        gguf: &mut GgufFile,
        prefix: &str,
        config: &LlamaConfig,
        cos: Tensor,
        sin: Tensor,
        device: &Device,
    ) -> ModelResult<Self> {
        let self_attn = QuantizedQwen2Attention::load(gguf, prefix, config, cos, sin, device)?;
        let mlp = QuantizedLlamaMLP::load(gguf, prefix, device)?;

        let input_ln_weight = gguf
            .tensor(&format!("{prefix}.attn_norm.weight"), device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize norm: {e}")))?;
        let input_layernorm = RmsNorm::new(input_ln_weight, config.rms_norm_eps);

        let post_ln_weight = gguf
            .tensor(&format!("{prefix}.ffn_norm.weight"), device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize norm: {e}")))?;
        let post_attention_layernorm = RmsNorm::new(post_ln_weight, config.rms_norm_eps);

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
        residual: Option<&Tensor>,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<(Tensor, Tensor)> {
        let (normed, residual) = if let Some(residual) = residual {
            crate::ops::fused_add_rms_norm(hidden_states, residual, &self.input_layernorm)
                .map_err(ModelError::Candle)?
        } else {
            let normed = crate::ops::rms_norm(hidden_states, &self.input_layernorm)
                .map_err(ModelError::Candle)?;
            (normed, hidden_states.clone())
        };

        let attn_output = self.self_attn.forward(&normed, positions, kv_cache)?;

        let (normed, residual) =
            crate::ops::fused_add_rms_norm(&attn_output, &residual, &self.post_attention_layernorm)
                .map_err(ModelError::Candle)?;

        let mlp_output = self.mlp.forward(&normed).map_err(ModelError::Candle)?;

        Ok((mlp_output, residual))
    }
}

// ---------------------------------------------------------------------------
// QuantizedQwen2Model
// ---------------------------------------------------------------------------

struct QuantizedQwen2Model {
    embed_tokens: Embedding,
    layers: Vec<QuantizedQwen2DecoderLayer>,
    norm: RmsNorm,
}

impl QuantizedQwen2Model {
    fn load(gguf: &mut GgufFile, config: &LlamaConfig, device: &Device) -> ModelResult<Self> {
        let embed_weight = gguf
            .tensor("token_embd.weight", device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize embedding: {e}")))?;
        let embed_tokens = Embedding::new(embed_weight);

        let (cos, sin) = precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = QuantizedQwen2DecoderLayer::load(
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
        let norm = RmsNorm::new(norm_weight, config.rms_norm_eps);

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

        let mut residual: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_handle = kv_cache.as_mut().map(|s| s.layer_handle(i));
            let (hs, res) =
                layer.forward(&hidden_states, residual.as_ref(), positions, layer_handle)?;
            hidden_states = hs;
            residual = Some(res);
        }

        let (normed, _) =
            crate::ops::fused_add_rms_norm(&hidden_states, residual.as_ref().unwrap(), &self.norm)
                .map_err(ModelError::Candle)?;
        Ok(normed)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// QuantizedQwen2ForCausalLM
// ---------------------------------------------------------------------------

pub struct QuantizedQwen2ForCausalLM {
    model: QuantizedQwen2Model,
    lm_head: QuantizedLinear,
}

impl QuantizedQwen2ForCausalLM {
    pub fn load(gguf: &mut GgufFile, config: &LlamaConfig, device: &Device) -> ModelResult<Self> {
        let model = QuantizedQwen2Model::load(gguf, config, device)?;

        let has_output = gguf.tensor_names().contains(&"output.weight");
        let lm_head = if has_output {
            QuantizedLinear::from_gguf(gguf, "output.weight", device)?
        } else {
            let qt = gguf.tensor("token_embd.weight", device)?;
            QuantizedLinear::from_qtensor(qt)?
        };

        Ok(Self { model, lm_head })
    }
}

impl crate::Model for QuantizedQwen2ForCausalLM {
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
// Factory function for the GGUF registry
// ---------------------------------------------------------------------------

/// Create a quantized Qwen2 model from a GGUF file.
pub fn create_qwen2_gguf(
    gguf: &mut GgufFile,
    config: &HfModelConfig,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let qwen2_config = Qwen2Config::from_hf_config(config)?;
    let model = QuantizedQwen2ForCausalLM::load(gguf, &qwen2_config.0, device)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen2_gguf_registry() {
        let registry = crate::ModelRegistry::default_registry();
        assert!(registry.contains_gguf("qwen2"));
    }

    #[test]
    fn test_qwen3_gguf_registry() {
        let registry = crate::ModelRegistry::default_registry();
        assert!(registry.contains_gguf("qwen3"));
        assert!(registry.contains_gguf("qwen35"));
    }
}
