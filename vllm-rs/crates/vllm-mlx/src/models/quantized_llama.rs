// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Quantized LLaMA model architecture for MLX.
//!
//! Loads mlx-community pre-quantized safetensors (4-bit group quantization):
//! `nn::QuantizedLinear` for Q/K/V/O projections and MLP linear layers,
//! `nn::QuantizedEmbedding` for embed_tokens and optionally lm_head.
//! Norms stay `nn::RmsNorm` (f32). RoPE is unchanged (half-split HF convention).
//!
//! Weight format (per linear layer):
//! - `*.weight` — packed u32 quantized weights
//! - `*.scales` — f16 quantization scales
//! - `*.biases` — f16 quantization biases

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use crate::cache::MlxKvCache;
use crate::models::llama::{LlamaConfig, load_safetensors_weights};
use vllm_model::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// Quantization config
// ---------------------------------------------------------------------------

/// Quantization parameters parsed from config.json.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub group_size: i32,
    pub bits: i32,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            group_size: 64,
            bits: 4,
        }
    }
}

impl QuantConfig {
    /// Parse from the `"quantization"` field in config.json extras.
    pub fn from_hf_config(config: &HfModelConfig) -> Option<Self> {
        let quant = config.extra.get("quantization")?;
        let obj = quant.as_object()?;
        let group_size = obj.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32;
        let bits = obj.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32;
        Some(Self { group_size, bits })
    }
}

// ---------------------------------------------------------------------------
// Helpers: construct quantized layers from loaded weights
// ---------------------------------------------------------------------------

/// Construct a `QuantizedLinear` directly from loaded weight arrays.
/// Avoids the builder's wasteful random-init + quantize step.
pub(crate) fn make_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> nn::QuantizedLinear {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("Weight not found: {prefix}.weight");
            Array::from_f32(0.0)
        });
    let scales = weights
        .get(&format!("{prefix}.scales"))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("Weight not found: {prefix}.scales");
            Array::from_f32(0.0)
        });
    let biases = weights
        .get(&format!("{prefix}.biases"))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("Weight not found: {prefix}.biases");
            Array::from_f32(0.0)
        });

    nn::QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner: nn::Linear {
            weight: Param::new(weight),
            bias: Param::new(None),
        },
    }
}

/// Construct a `QuantizedEmbedding` directly from loaded weight arrays.
pub(crate) fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> nn::QuantizedEmbedding {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("Weight not found: {prefix}.weight");
            Array::from_f32(0.0)
        });
    let scales = weights
        .get(&format!("{prefix}.scales"))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("Weight not found: {prefix}.scales");
            Array::from_f32(0.0)
        });
    let biases = weights
        .get(&format!("{prefix}.biases"))
        .cloned()
        .unwrap_or_else(|| {
            tracing::warn!("Weight not found: {prefix}.biases");
            Array::from_f32(0.0)
        });

    nn::QuantizedEmbedding {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner: nn::Embedding {
            weight: Param::new(weight),
        },
    }
}

/// Look up a weight by name and assign it.
pub(crate) fn assign_weight(
    target: &mut Param<Array>,
    weights: &HashMap<String, Array>,
    name: &str,
) {
    if let Some(w) = weights.get(name) {
        target.value = w.clone();
    } else {
        tracing::warn!("Weight not found: {name}");
    }
}

// ---------------------------------------------------------------------------
// MlxEmbedTokens / MlxLmHead — handle mixed quantized/float weights
// ---------------------------------------------------------------------------

/// Embedding that may or may not be quantized.
///
/// mlx-community quantized models often leave the embedding layer as float16
/// (no scales/biases). This enum dispatches to the right implementation.
pub(crate) enum MlxEmbedTokens {
    Quantized(nn::QuantizedEmbedding),
    Float(nn::Embedding),
}

impl MlxEmbedTokens {
    /// Create from loaded weights. Uses `QuantizedEmbedding` if `{prefix}.scales`
    /// exists, otherwise falls back to a regular float `Embedding`.
    pub(crate) fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Self {
        if weights.contains_key(&format!("{prefix}.scales")) {
            Self::Quantized(make_quantized_embedding(weights, prefix, group_size, bits))
        } else if let Some(w) = weights.get(&format!("{prefix}.weight")) {
            tracing::info!("Embedding at {prefix} is not quantized, using float");
            let mut emb =
                nn::Embedding::new(w.dim(0), w.dim(1)).expect("failed to create Embedding");
            emb.weight.value = w.clone();
            Self::Float(emb)
        } else {
            tracing::warn!("Embedding weights not found at {prefix}, creating quantized stub");
            Self::Quantized(make_quantized_embedding(weights, prefix, group_size, bits))
        }
    }

    pub(crate) fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::Quantized(emb) => emb.forward(x),
            Self::Float(emb) => emb.forward(x),
        }
    }

    pub(crate) fn as_linear(&self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::Quantized(emb) => emb.as_linear(x),
            Self::Float(emb) => emb.as_linear(x),
        }
    }
}

/// LM head that may or may not be quantized.
///
/// Same rationale as `MlxEmbedTokens` — some quantized models leave lm_head
/// as float.
pub(crate) enum MlxLmHead {
    Quantized(nn::QuantizedLinear),
    Float(nn::Linear),
}

impl MlxLmHead {
    /// Create from loaded weights. Uses `QuantizedLinear` if `{prefix}.scales`
    /// exists, otherwise falls back to a regular float `Linear`.
    pub(crate) fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Self {
        if weights.contains_key(&format!("{prefix}.scales")) {
            Self::Quantized(make_quantized_linear(weights, prefix, group_size, bits))
        } else if let Some(w) = weights.get(&format!("{prefix}.weight")) {
            tracing::info!("LM head at {prefix} is not quantized, using float");
            Self::Float(nn::Linear {
                weight: Param::new(w.clone()),
                bias: Param::new(None),
            })
        } else {
            tracing::warn!("LM head weights not found at {prefix}, creating quantized stub");
            Self::Quantized(make_quantized_linear(weights, prefix, group_size, bits))
        }
    }

    pub(crate) fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::Quantized(lin) => lin.forward(x),
            Self::Float(lin) => lin.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedLlamaMLP
// ---------------------------------------------------------------------------

/// Quantized LLaMA MLP (SiLU-gated feed-forward network).
pub struct MlxQuantizedLlamaMLP {
    gate_proj: nn::QuantizedLinear,
    up_proj: nn::QuantizedLinear,
    down_proj: nn::QuantizedLinear,
}

impl MlxQuantizedLlamaMLP {
    /// Create from loaded weights.
    pub fn from_weights(weights: &HashMap<String, Array>, prefix: &str, qc: &QuantConfig) -> Self {
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

    /// Forward pass: gate_proj(x) → SiLU → * up_proj(x) → down_proj
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedLlamaAttention
// ---------------------------------------------------------------------------

/// Quantized LLaMA multi-head attention with RoPE and optional GQA.
pub struct MlxQuantizedLlamaAttention {
    q_proj: nn::QuantizedLinear,
    k_proj: nn::QuantizedLinear,
    v_proj: nn::QuantizedLinear,
    o_proj: nn::QuantizedLinear,
    /// Optional per-head Q norm (Qwen3 uses this).
    q_norm: Option<nn::RmsNorm>,
    /// Optional per-head K norm (Qwen3 uses this).
    k_norm: Option<nn::RmsNorm>,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
}

impl MlxQuantizedLlamaAttention {
    /// Create from loaded weights.
    pub fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &LlamaConfig,
        qc: &QuantConfig,
    ) -> Self {
        // Optional QK norms (Qwen3 has per-head q_norm and k_norm).
        let q_norm = weights
            .get(&format!("{prefix}.q_norm.weight"))
            .and_then(|w| {
                let head_dim = w.dim(0);
                let mut norm = nn::RmsNormBuilder::new(head_dim).eps(1e-6).build().ok()?;
                norm.weight.value = w.clone();
                Some(norm)
            });
        let k_norm = weights
            .get(&format!("{prefix}.k_norm.weight"))
            .and_then(|w| {
                let head_dim = w.dim(0);
                let mut norm = nn::RmsNormBuilder::new(head_dim).eps(1e-6).build().ok()?;
                norm.weight.value = w.clone();
                Some(norm)
            });

        Self {
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
                r.base = config.rope_theta;
                r
            },
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            sliding_window: config.sliding_window,
        }
    }

    /// Forward pass.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        // Q/K/V projections.
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape: [seq, hidden] -> [seq, heads, head_dim]
        let q = q.reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?;
        let k = k.reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?;

        // Apply optional per-head QK norms (Qwen3).
        let q = if let Some(ref mut norm) = self.q_norm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref mut norm) = self.k_norm {
            norm.forward(&k)?
        } else {
            k
        };

        // [seq, heads, head_dim] -> [1, heads, seq, head_dim]
        let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut v = v
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // RoPE: uses positions[0] as offset for proper positional encoding.
        let offset = if positions.size() > 0 {
            positions.reshape(&[-1])?.min(None)?.item::<i32>()
        } else {
            0
        };
        let q = self.rope.forward((&q, offset))?;
        k = self.rope.forward((&k, offset))?;

        // KV cache update (lazy — no actual concat until eval).
        if let Some((ck, cv)) = cache.take() {
            k = concatenate_axis(&[ck, k], 2)?;
            v = concatenate_axis(&[cv, v], 2)?;
        }
        // Store the full cache (for future steps).
        *cache = Some((k.clone(), v.clone()));

        // Sliding window: trim K/V to only the last `w` positions.
        if let Some(w) = self.sliding_window {
            let kv_len = k.dim(2) as usize;
            if kv_len > w {
                let start = (kv_len - w) as i32;
                let end = kv_len as i32;
                k = k.try_index((.., .., start..end, ..))?;
                v = v.try_index((.., .., start..end, ..))?;
            }
        }

        // Fused SDPA.
        let mask = if seq_len > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;

        // out: [1, heads, seq, head_dim] -> [seq, hidden]
        let hidden = (self.num_heads * self.head_dim) as i32;
        let out = out
            .squeeze_axes(&[0])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq_len, hidden])?;

        self.o_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedLlamaDecoderLayer
// ---------------------------------------------------------------------------

/// A single quantized LLaMA decoder layer.
pub struct MlxQuantizedLlamaDecoderLayer {
    self_attn: MlxQuantizedLlamaAttention,
    mlp: MlxQuantizedLlamaMLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

impl MlxQuantizedLlamaDecoderLayer {
    /// Create from loaded weights.
    pub fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &LlamaConfig,
        qc: &QuantConfig,
    ) -> Result<Self, Exception> {
        let mut input_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        let mut post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;

        assign_weight(
            &mut input_layernorm.weight,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        );
        assign_weight(
            &mut post_attention_layernorm.weight,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        );

        Ok(Self {
            self_attn: MlxQuantizedLlamaAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                qc,
            ),
            mlp: MlxQuantizedLlamaMLP::from_weights(weights, &format!("{prefix}.mlp"), qc),
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, positions, cache)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        hidden_states.add(&mlp_output)
    }
}

// ---------------------------------------------------------------------------
// MlxQuantizedLlamaForCausalLM
// ---------------------------------------------------------------------------

/// Quantized LLaMA for causal language modeling using MLX.
///
/// Uses `nn::QuantizedLinear` for projections and MLP.
/// Embedding and lm_head may be float if the quantized model didn't quantize them.
pub struct MlxQuantizedLlamaForCausalLM {
    embed_tokens: MlxEmbedTokens,
    layers: Vec<MlxQuantizedLlamaDecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<MlxLmHead>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: LlamaConfig,
}

impl MlxQuantizedLlamaForCausalLM {
    /// Load model directly from safetensors weights.
    pub fn load(
        model_dir: &Path,
        config: &LlamaConfig,
        qc: &QuantConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weights = load_safetensors_weights(model_dir)?;

        // Build embedding (auto-detects quantized vs float).
        let embed_tokens =
            MlxEmbedTokens::from_weights(&weights, "model.embed_tokens", qc.group_size, qc.bits);

        // Build decoder layers.
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQuantizedLlamaDecoderLayer::from_weights(
                &weights,
                &format!("model.layers.{i}"),
                config,
                qc,
            )?);
        }

        // Build final norm.
        let mut norm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        assign_weight(&mut norm.weight, &weights, "model.norm.weight");

        // Build lm_head (auto-detects quantized vs float).
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(MlxLmHead::from_weights(
                &weights,
                "lm_head",
                qc.group_size,
                qc.bits,
            ))
        };

        // Eval to materialize loaded weights.
        mlx_rs::transforms::eval(weights.values())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            tie_word_embeddings: config.tie_word_embeddings,
            config: config.clone(),
        })
    }
}

impl super::MlxModel for MlxQuantizedLlamaForCausalLM {
    fn forward(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> mlx_rs::error::Result<Array> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i])?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        // Compute logits.
        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };

        // Cast logits to f32 for sampling.
        logits.as_dtype(Dtype::Float32)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

/// Factory function for creating a quantized MLX LLaMA model.
pub fn create_mlx_quantized_llama(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let llama_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    tracing::info!(
        "Loading quantized MLX LLaMA (group_size={}, bits={})",
        qc.group_size,
        qc.bits
    );

    let model = MlxQuantizedLlamaForCausalLM::load(model_dir, &llama_config, &qc, dtype)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::builder::Builder;

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
            tie_word_embeddings: true,
            sliding_window: None,
        }
    }

    fn test_quant_config() -> QuantConfig {
        QuantConfig {
            group_size: 32,
            bits: 4,
        }
    }

    /// Build a quantized model with random-initialized weights for testing.
    /// Uses the builder approach (quantizes random data) since we have no
    /// pre-quantized safetensors in tests.
    fn build_test_model(
        config: &LlamaConfig,
        qc: &QuantConfig,
    ) -> Result<MlxQuantizedLlamaForCausalLM, Exception> {
        let embed_tokens = MlxEmbedTokens::Quantized(
            nn::QuantizedEmbeddingBuilder::new(config.vocab_size as i32, config.hidden_size as i32)
                .group_size(qc.group_size)
                .bits(qc.bits)
                .build()?,
        );

        let hidden = config.hidden_size as i32;
        let intermediate = config.intermediate_size as i32;
        let q_size = (config.num_attention_heads * config.head_dim) as i32;
        let kv_size = (config.num_kv_heads * config.head_dim) as i32;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            let self_attn = MlxQuantizedLlamaAttention {
                q_proj: nn::QuantizedLinearBuilder::new(hidden, q_size)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?,
                k_proj: nn::QuantizedLinearBuilder::new(hidden, kv_size)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?,
                v_proj: nn::QuantizedLinearBuilder::new(hidden, kv_size)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?,
                o_proj: nn::QuantizedLinearBuilder::new(q_size, hidden)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?,
                q_norm: None,
                k_norm: None,
                rope: {
                    let mut r = nn::Rope::new(config.head_dim as i32);
                    r.base = config.rope_theta;
                    r
                },
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_kv_heads,
                head_dim: config.head_dim,
                scale: 1.0 / (config.head_dim as f32).sqrt(),
                sliding_window: config.sliding_window,
            };

            let mlp = MlxQuantizedLlamaMLP {
                gate_proj: nn::QuantizedLinearBuilder::new(hidden, intermediate)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?,
                up_proj: nn::QuantizedLinearBuilder::new(hidden, intermediate)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?,
                down_proj: nn::QuantizedLinearBuilder::new(intermediate, hidden)
                    .group_size(qc.group_size)
                    .bits(qc.bits)
                    .bias(false)
                    .build()?,
            };

            layers.push(MlxQuantizedLlamaDecoderLayer {
                self_attn,
                mlp,
                input_layernorm: nn::RmsNormBuilder::new(hidden)
                    .eps(config.rms_norm_eps)
                    .build()?,
                post_attention_layernorm: nn::RmsNormBuilder::new(hidden)
                    .eps(config.rms_norm_eps)
                    .build()?,
            });
        }

        Ok(MlxQuantizedLlamaForCausalLM {
            embed_tokens,
            layers,
            norm: nn::RmsNormBuilder::new(hidden)
                .eps(config.rms_norm_eps)
                .build()?,
            lm_head: None,
            tie_word_embeddings: config.tie_word_embeddings,
            config: config.clone(),
        })
    }

    #[test]
    fn test_quant_config_parse() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "quantization": { "group_size": 64, "bits": 4 }
            }"#,
        )
        .unwrap();

        let qc = QuantConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(qc.group_size, 64);
        assert_eq!(qc.bits, 4);
    }

    #[test]
    fn test_quant_config_missing() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "intermediate_size": 11008,
                "vocab_size": 32000
            }"#,
        )
        .unwrap();

        assert!(QuantConfig::from_hf_config(&hf_config).is_none());
    }

    #[test]
    fn test_quantized_llama_forward() {
        let config = test_config();
        let qc = test_quant_config();
        let mut model = build_test_model(&config, &qc).unwrap();

        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        let logits = <MlxQuantizedLlamaForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
        )
        .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);
    }

    #[test]
    fn test_quantized_llama_prefill_and_decode() {
        let config = test_config();
        let qc = test_quant_config();
        let mut model = build_test_model(&config, &qc).unwrap();
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        // Prefill: 3 tokens
        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let logits = <MlxQuantizedLlamaForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
        )
        .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);

        for entry in &kv_cache {
            assert!(entry.is_some());
        }

        // Decode: 1 token at position 3
        let decode_ids = Array::from_iter(vec![15i32], &[1]);
        let decode_pos = Array::from_iter(vec![3i32], &[1]);
        let logits2 = <MlxQuantizedLlamaForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &decode_ids,
            &decode_pos,
            &mut kv_cache,
        )
        .unwrap();
        logits2.eval().unwrap();
        assert_eq!(logits2.shape(), &[1, config.vocab_size as i32]);
    }
}
