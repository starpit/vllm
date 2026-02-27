// SPDX-License-Identifier: Apache-2.0
//! Quantized LLaMA model architecture using GGUF weights.
//!
//! Parallel to `llama.rs` but uses `QuantizedLinear` (wrapping `QMatMul`)
//! for linear projections. Non-linear layers (embedding, norms) are
//! dequantized to f32 since they are tiny relative to the linear layers.
//! KV cache remains in f32 (unquantized).

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::gguf::GgufFile;
use vllm_model::layers::{Embedding, QuantizedLinear, RmsNorm};
use vllm_model::weight::HfModelConfig;

use crate::attention::scaled_dot_product_attention;
use crate::llama::LlamaConfig;

// ---------------------------------------------------------------------------
// QuantizedLlamaMLP
// ---------------------------------------------------------------------------

/// LLaMA MLP with quantized linear projections.
struct QuantizedLlamaMLP {
    gate_proj: QuantizedLinear,
    up_proj: QuantizedLinear,
    down_proj: QuantizedLinear,
}

impl QuantizedLlamaMLP {
    fn load(gguf: &mut GgufFile, prefix: &str, device: &Device) -> ModelResult<Self> {
        Ok(Self {
            gate_proj: QuantizedLinear::from_gguf(
                gguf,
                &format!("{prefix}.ffn_gate.weight"),
                device,
            )?,
            up_proj: QuantizedLinear::from_gguf(gguf, &format!("{prefix}.ffn_up.weight"), device)?,
            down_proj: QuantizedLinear::from_gguf(
                gguf,
                &format!("{prefix}.ffn_down.weight"),
                device,
            )?,
        })
    }
}

impl Module for QuantizedLlamaMLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = gate.silu()?.mul(&up)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// QuantizedLlamaAttention
// ---------------------------------------------------------------------------

/// LLaMA attention with quantized Q/K/V/O projections.
///
/// Uses interleaved RoPE (GGML convention) rather than half-split RoPE
/// (HuggingFace convention) because GGUF weights are stored in GGML layout.
struct QuantizedLlamaAttention {
    q_proj: QuantizedLinear,
    k_proj: QuantizedLinear,
    v_proj: QuantizedLinear,
    o_proj: QuantizedLinear,
    /// Precomputed cos values: [max_position, head_dim/2]
    cos: Tensor,
    /// Precomputed sin values: [max_position, head_dim/2]
    sin: Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

/// Precompute cos/sin tables for interleaved RoPE (GGML convention).
///
/// Returns (cos, sin) each of shape `[max_position, head_dim/2]`.
fn precompute_freqs_cis(
    head_dim: usize,
    max_position: usize,
    rope_theta: f64,
    device: &Device,
) -> ModelResult<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| (1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device).map_err(ModelError::Candle)?;

    let positions: Vec<f32> = (0..max_position).map(|p| p as f32).collect();
    let positions = Tensor::new(positions.as_slice(), device).map_err(ModelError::Candle)?;

    let freqs = positions
        .reshape((max_position, 1))
        .map_err(ModelError::Candle)?
        .matmul(
            &inv_freq
                .reshape((1, half_dim))
                .map_err(ModelError::Candle)?,
        )
        .map_err(ModelError::Candle)?; // [max_position, half_dim]

    let cos = freqs.cos().map_err(ModelError::Candle)?;
    let sin = freqs.sin().map_err(ModelError::Candle)?;
    Ok((cos, sin))
}

/// Apply interleaved RoPE to a 3D tensor [seq_len, num_heads, head_dim].
///
/// Interleaved means adjacent pairs (x[2i], x[2i+1]) are rotated together:
///   y[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
///   y[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
fn apply_interleaved_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
) -> ModelResult<Tensor> {
    let (seq_len, num_heads, head_dim) = x.dims3().map_err(ModelError::Candle)?;
    let half_dim = head_dim / 2;

    // Gather cos/sin for positions: [seq_len, half_dim]
    let cos = cos.index_select(positions, 0).map_err(ModelError::Candle)?;
    let sin = sin.index_select(positions, 0).map_err(ModelError::Candle)?;

    // Reshape x to [seq_len, num_heads, half_dim, 2] for interleaved pairs.
    let x = x
        .reshape((seq_len, num_heads, half_dim, 2))
        .map_err(ModelError::Candle)?;
    let x0 = x
        .narrow(candle_core::D::Minus1, 0, 1)
        .map_err(ModelError::Candle)?
        .squeeze(candle_core::D::Minus1)
        .map_err(ModelError::Candle)?; // [seq_len, num_heads, half_dim]
    let x1 = x
        .narrow(candle_core::D::Minus1, 1, 1)
        .map_err(ModelError::Candle)?
        .squeeze(candle_core::D::Minus1)
        .map_err(ModelError::Candle)?;

    // cos/sin: [seq_len, half_dim] → [seq_len, 1, half_dim] for broadcasting.
    let cos = cos.unsqueeze(1).map_err(ModelError::Candle)?;
    let sin = sin.unsqueeze(1).map_err(ModelError::Candle)?;

    // y0 = x0*cos - x1*sin
    // y1 = x0*sin + x1*cos
    let y0 = (x0.broadcast_mul(&cos).map_err(ModelError::Candle)?
        - x1.broadcast_mul(&sin).map_err(ModelError::Candle)?)
    .map_err(ModelError::Candle)?;
    let y1 = (x0.broadcast_mul(&sin).map_err(ModelError::Candle)?
        + x1.broadcast_mul(&cos).map_err(ModelError::Candle)?)
    .map_err(ModelError::Candle)?;

    // Interleave y0, y1 back to [seq_len, num_heads, head_dim].
    let y0 = y0
        .unsqueeze(candle_core::D::Minus1)
        .map_err(ModelError::Candle)?; // [s, h, half, 1]
    let y1 = y1
        .unsqueeze(candle_core::D::Minus1)
        .map_err(ModelError::Candle)?;
    let y = Tensor::cat(&[&y0, &y1], candle_core::D::Minus1).map_err(ModelError::Candle)?; // [s, h, half, 2]
    y.reshape((seq_len, num_heads, head_dim))
        .map_err(ModelError::Candle)
}

impl QuantizedLlamaAttention {
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

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            cos,
            sin,
            num_q_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: 1.0 / (config.head_dim as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<Tensor> {
        let num_tokens = hidden_states.dim(0).map_err(ModelError::Candle)?;

        // Q/K/V projections (QMatMul output is f32 on CPU).
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

        // Apply interleaved RoPE (GGML convention).
        let q = apply_interleaved_rope(&q, &self.cos, &self.sin, positions)?;
        let k = apply_interleaved_rope(&k, &self.cos, &self.sin, positions)?;

        // Merge with KV cache.
        let (k_full, v_full) = if let Some(mut handle) = kv_cache {
            if let Some((cached_k, cached_v)) = handle.take_cached()? {
                let k_cat = Tensor::cat(&[&cached_k, &k], 0).map_err(ModelError::Candle)?;
                let v_cat = Tensor::cat(&[&cached_v, &v], 0).map_err(ModelError::Candle)?;
                handle.store(k_cat.clone(), v_cat.clone())?;
                (k_cat, v_cat)
            } else {
                handle.store(k.clone(), v.clone())?;
                (k, v)
            }
        } else {
            (k, v)
        };

        // Scaled dot-product attention.
        let attn_output = scaled_dot_product_attention(&q, &k_full, &v_full, self.scale)?;

        // Reshape and output projection.
        let attn_output = attn_output
            .reshape((num_tokens, self.num_q_heads * self.head_dim))
            .map_err(ModelError::Candle)?;

        self.o_proj
            .forward(&attn_output)
            .map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// QuantizedLlamaDecoderLayer
// ---------------------------------------------------------------------------

/// A single quantized LLaMA decoder layer.
struct QuantizedLlamaDecoderLayer {
    self_attn: QuantizedLlamaAttention,
    mlp: QuantizedLlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedLlamaDecoderLayer {
    fn load(
        gguf: &mut GgufFile,
        prefix: &str,
        config: &LlamaConfig,
        cos: Tensor,
        sin: Tensor,
        device: &Device,
    ) -> ModelResult<Self> {
        let self_attn = QuantizedLlamaAttention::load(gguf, prefix, config, cos, sin, device)?;
        let mlp = QuantizedLlamaMLP::load(gguf, prefix, device)?;

        // Norms: dequantize the QTensor to a regular f32 Tensor.
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
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
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
// QuantizedLlamaModel
// ---------------------------------------------------------------------------

/// Quantized LLaMA transformer backbone.
struct QuantizedLlamaModel {
    embed_tokens: Embedding,
    layers: Vec<QuantizedLlamaDecoderLayer>,
    norm: RmsNorm,
}

impl QuantizedLlamaModel {
    fn load(gguf: &mut GgufFile, config: &LlamaConfig, device: &Device) -> ModelResult<Self> {
        // Embedding: dequantize to f32 (it's relatively small).
        let embed_weight = gguf
            .tensor("token_embd.weight", device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize embedding: {e}")))?;
        let embed_tokens = Embedding::new(embed_weight);

        // Precompute interleaved RoPE cos/sin tables (shared across layers).
        let (cos, sin) = precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;

        // Decoder layers.
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = QuantizedLlamaDecoderLayer::load(
                gguf,
                &format!("blk.{i}"),
                config,
                cos.clone(),
                sin.clone(),
                device,
            )?;
            layers.push(layer);
        }

        // Final norm: dequantize.
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
// QuantizedLlamaForCausalLM
// ---------------------------------------------------------------------------

/// Quantized LLaMA for causal language modeling.
pub struct QuantizedLlamaForCausalLM {
    model: QuantizedLlamaModel,
    lm_head: QuantizedLinear,
    tie_word_embeddings: bool,
    /// Dequantized embedding weight for tied embeddings fallback.
    embed_weight: Option<Tensor>,
}

impl QuantizedLlamaForCausalLM {
    /// Load the full quantized model from a GGUF file.
    pub fn load(gguf: &mut GgufFile, config: &LlamaConfig, device: &Device) -> ModelResult<Self> {
        let model = QuantizedLlamaModel::load(gguf, config, device)?;

        // lm_head: some GGUF models have "output.weight", others tie with embedding.
        let has_output = gguf.tensor_names().contains(&"output.weight");
        let (lm_head, tie_word_embeddings, embed_weight) = if has_output {
            let lm = QuantizedLinear::from_gguf(gguf, "output.weight", device)?;
            (lm, false, None)
        } else {
            // Tie with embedding: use the embedding weight as lm_head.
            // We need to create a QMatMul from the embedding. Since we already
            // dequantized the embedding, we create a QTensor from f32.
            // Actually, re-read the original quantized tensor.
            let qt = gguf.tensor("token_embd.weight", device)?;
            let lm = QuantizedLinear::from_qtensor(qt)?;
            (lm, true, None)
        };

        Ok(Self {
            model,
            lm_head,
            tie_word_embeddings,
            embed_weight,
        })
    }
}

impl crate::Model for QuantizedLlamaForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self.model.forward(input_ids, positions, kv_cache)?;
        let logits = if self.tie_word_embeddings {
            if let Some(ref w) = self.embed_weight {
                // Use dequantized embedding weight directly.
                hidden_states
                    .matmul(&w.t().map_err(ModelError::Candle)?)
                    .map_err(ModelError::Candle)?
            } else {
                self.lm_head
                    .forward(&hidden_states)
                    .map_err(ModelError::Candle)?
            }
        } else {
            self.lm_head
                .forward(&hidden_states)
                .map_err(ModelError::Candle)?
        };
        // Cast logits to f32 for sampling.
        logits.to_dtype(DType::F32).map_err(ModelError::Candle)
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }
}

// ---------------------------------------------------------------------------
// Factory function for the GGUF registry
// ---------------------------------------------------------------------------

/// Create a quantized LLaMA model from a GGUF file.
pub fn create_llama_gguf(
    gguf: &mut GgufFile,
    config: &HfModelConfig,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let llama_config = LlamaConfig::from_hf_config(config)?;
    let model = QuantizedLlamaForCausalLM::load(gguf, &llama_config, device)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_config_from_hf() {
        // Verify config parsing works for the quantized path too.
        let hf_config = HfModelConfig {
            hidden_size: Some(4096),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            num_hidden_layers: Some(32),
            intermediate_size: Some(11008),
            vocab_size: Some(32000),
            max_position_embeddings: Some(4096),
            rms_norm_eps: Some(1e-5),
            rope_theta: Some(10000.0),
            ..Default::default()
        };
        let config = LlamaConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
    }
}
