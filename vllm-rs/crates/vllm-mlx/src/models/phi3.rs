// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Phi-3 model architecture for MLX.
//!
//! Phi-3 is similar to LLaMA but uses fused projections:
//! - `qkv_proj` — fused Q/K/V projection (split after projection)
//! - `gate_up_proj` — fused gate+up projection (split after projection)
//! - `o_proj`, `down_proj` — separate (same as LLaMA)
//!
//! Activation: SiLU (same as LLaMA).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use crate::cache::MlxKvCache;
use crate::models::llama::{LlamaConfig, assign_weight, load_safetensors_weights};
use crate::models::quantized_llama::{
    MlxEmbedTokens, MlxLmHead, QuantConfig, make_quantized_linear,
};
use vllm_model::weight::HfModelConfig;

// ---------------------------------------------------------------------------
// MlxPhi3MLP — fused gate_up_proj
// ---------------------------------------------------------------------------

struct MlxPhi3MLP {
    gate_up_proj: nn::Linear,
    down_proj: nn::Linear,
    intermediate_size: usize,
}

impl MlxPhi3MLP {
    fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_up_proj: nn::LinearBuilder::new(hidden_size, 2 * intermediate_size)
                .bias(false)
                .build()?,
            down_proj: nn::LinearBuilder::new(intermediate_size, hidden_size)
                .bias(false)
                .build()?,
            intermediate_size: intermediate_size as usize,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.gate_up_proj.weight,
            weights,
            &format!("{prefix}.gate_up_proj.weight"),
        );
        assign_weight(
            &mut self.down_proj.weight,
            weights,
            &format!("{prefix}.down_proj.weight"),
        );
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate_up = self.gate_up_proj.forward(x)?;
        // Split along last dim: [seq, 2*intermediate] → gate [seq, intermediate], up [seq, intermediate]
        let parts = gate_up.split_axis(&[self.intermediate_size as i32], -1)?;
        let gate = nn::silu(&parts[0])?;
        let hidden = gate.multiply(&parts[1])?;
        self.down_proj.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// MlxPhi3Attention — fused qkv_proj
// ---------------------------------------------------------------------------

struct MlxPhi3Attention {
    qkv_proj: nn::Linear,
    o_proj: nn::Linear,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    q_size: usize,
    kv_size: usize,
}

impl MlxPhi3Attention {
    fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let q_size = config.num_attention_heads * config.head_dim;
        let kv_size = config.num_kv_heads * config.head_dim;
        let qkv_size = (q_size + 2 * kv_size) as i32;

        Ok(Self {
            qkv_proj: nn::LinearBuilder::new(hidden, qkv_size)
                .bias(false)
                .build()?,
            o_proj: nn::LinearBuilder::new(q_size as i32, hidden)
                .bias(false)
                .build()?,
            rope: {
                let mut r = nn::Rope::new(config.head_dim as i32);
                r.base = config.rope_theta;
                r
            },
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            q_size,
            kv_size,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.qkv_proj.weight,
            weights,
            &format!("{prefix}.qkv_proj.weight"),
        );
        assign_weight(
            &mut self.o_proj.weight,
            weights,
            &format!("{prefix}.o_proj.weight"),
        );
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        // Fused QKV projection.
        let qkv = self.qkv_proj.forward(hidden_states)?;
        // Split: [seq, q_size + kv_size + kv_size]
        let parts = qkv.split_axis(
            &[self.q_size as i32, (self.q_size + self.kv_size) as i32],
            -1,
        )?;

        // Reshape: [seq, size] -> [1, heads, seq, head_dim]
        let q = parts[0]
            .reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let mut k = parts[1]
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let mut v = parts[2]
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // RoPE
        let offset = if positions.size() > 0 {
            positions.reshape(&[-1])?.min(None)?.item::<i32>()
        } else {
            0
        };
        let q = self.rope.forward((&q, offset))?;
        k = self.rope.forward((&k, offset))?;

        // KV cache
        if let Some((ck, cv)) = cache.take() {
            k = concatenate_axis(&[ck, k], 2)?;
            v = concatenate_axis(&[cv, v], 2)?;
        }
        *cache = Some((k.clone(), v.clone()));

        // SDPA
        let mask = if seq_len > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;

        let hidden = (self.num_heads * self.head_dim) as i32;
        let out = out
            .squeeze_axes(&[0])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq_len, hidden])?;

        self.o_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// MlxPhi3DecoderLayer
// ---------------------------------------------------------------------------

struct MlxPhi3DecoderLayer {
    self_attn: MlxPhi3Attention,
    mlp: MlxPhi3MLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

impl MlxPhi3DecoderLayer {
    fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: MlxPhi3Attention::new(config)?,
            mlp: MlxPhi3MLP::new(config.hidden_size as i32, config.intermediate_size as i32)?,
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        self.self_attn
            .load_weights(weights, &format!("{prefix}.self_attn"));
        self.mlp.load_weights(weights, &format!("{prefix}.mlp"));
        assign_weight(
            &mut self.input_layernorm.weight,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        );
        assign_weight(
            &mut self.post_attention_layernorm.weight,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        );
    }

    fn forward(
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
// MlxPhi3ForCausalLM (float)
// ---------------------------------------------------------------------------

pub struct MlxPhi3ForCausalLM {
    embed_tokens: nn::Embedding,
    layers: Vec<MlxPhi3DecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<nn::Linear>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: LlamaConfig,
}

impl MlxPhi3ForCausalLM {
    fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(MlxPhi3DecoderLayer::new(config)?);
        }

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size as i32, config.vocab_size as i32)
                    .bias(false)
                    .build()?,
            )
        };

        Ok(Self {
            embed_tokens: nn::Embedding::new(config.vocab_size as i32, config.hidden_size as i32)?,
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            lm_head,
            tie_word_embeddings: config.tie_word_embeddings,
            config: config.clone(),
        })
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>) {
        assign_weight(
            &mut self.embed_tokens.weight,
            weights,
            "model.embed_tokens.weight",
        );
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("model.layers.{i}"));
        }
        assign_weight(&mut self.norm.weight, weights, "model.norm.weight");
        if let Some(ref mut lm_head) = self.lm_head {
            assign_weight(&mut lm_head.weight, weights, "lm_head.weight");
        }
    }

    fn load(
        model_dir: &Path,
        config: &LlamaConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut model = Self::new(config)?;
        let weights = load_safetensors_weights(model_dir)?;
        model.load_weights(&weights);
        mlx_rs::transforms::eval(weights.values())?;
        Ok(model)
    }
}

impl super::MlxModel for MlxPhi3ForCausalLM {
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

        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };

        logits.as_dtype(Dtype::Float32)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// Quantized Phi-3
// ---------------------------------------------------------------------------

struct MlxQuantizedPhi3MLP {
    gate_up_proj: nn::QuantizedLinear,
    down_proj: nn::QuantizedLinear,
    intermediate_size: usize,
}

impl MlxQuantizedPhi3MLP {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        qc: &QuantConfig,
        intermediate_size: usize,
    ) -> Self {
        Self {
            gate_up_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.gate_up_proj"),
                qc.group_size,
                qc.bits,
            ),
            down_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.down_proj"),
                qc.group_size,
                qc.bits,
            ),
            intermediate_size,
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let parts = gate_up.split_axis(&[self.intermediate_size as i32], -1)?;
        let gate = nn::silu(&parts[0])?;
        let hidden = gate.multiply(&parts[1])?;
        self.down_proj.forward(&hidden)
    }
}

struct MlxQuantizedPhi3Attention {
    qkv_proj: nn::QuantizedLinear,
    o_proj: nn::QuantizedLinear,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    q_size: usize,
    kv_size: usize,
}

impl MlxQuantizedPhi3Attention {
    fn from_weights(
        weights: &HashMap<String, Array>,
        prefix: &str,
        config: &LlamaConfig,
        qc: &QuantConfig,
    ) -> Self {
        Self {
            qkv_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.qkv_proj"),
                qc.group_size,
                qc.bits,
            ),
            o_proj: make_quantized_linear(
                weights,
                &format!("{prefix}.o_proj"),
                qc.group_size,
                qc.bits,
            ),
            rope: {
                let mut r = nn::Rope::new(config.head_dim as i32);
                r.base = config.rope_theta;
                r
            },
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            q_size: config.num_attention_heads * config.head_dim,
            kv_size: config.num_kv_heads * config.head_dim,
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        let qkv = self.qkv_proj.forward(hidden_states)?;
        let parts = qkv.split_axis(
            &[self.q_size as i32, (self.q_size + self.kv_size) as i32],
            -1,
        )?;

        let q = parts[0]
            .reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let mut k = parts[1]
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
        let mut v = parts[2]
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        let offset = if positions.size() > 0 {
            positions.reshape(&[-1])?.min(None)?.item::<i32>()
        } else {
            0
        };
        let q = self.rope.forward((&q, offset))?;
        k = self.rope.forward((&k, offset))?;

        if let Some((ck, cv)) = cache.take() {
            k = concatenate_axis(&[ck, k], 2)?;
            v = concatenate_axis(&[cv, v], 2)?;
        }
        *cache = Some((k.clone(), v.clone()));

        let mask = if seq_len > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;

        let hidden = (self.num_heads * self.head_dim) as i32;
        let out = out
            .squeeze_axes(&[0])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq_len, hidden])?;

        self.o_proj.forward(&out)
    }
}

struct MlxQuantizedPhi3DecoderLayer {
    self_attn: MlxQuantizedPhi3Attention,
    mlp: MlxQuantizedPhi3MLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

impl MlxQuantizedPhi3DecoderLayer {
    fn from_weights(
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

        crate::models::quantized_llama::assign_weight(
            &mut input_layernorm.weight,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        );
        crate::models::quantized_llama::assign_weight(
            &mut post_attention_layernorm.weight,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        );

        Ok(Self {
            self_attn: MlxQuantizedPhi3Attention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                qc,
            ),
            mlp: MlxQuantizedPhi3MLP::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                qc,
                config.intermediate_size,
            ),
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
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

pub struct MlxQuantizedPhi3ForCausalLM {
    embed_tokens: MlxEmbedTokens,
    layers: Vec<MlxQuantizedPhi3DecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: Option<MlxLmHead>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: LlamaConfig,
}

impl MlxQuantizedPhi3ForCausalLM {
    fn load(
        model_dir: &Path,
        config: &LlamaConfig,
        qc: &QuantConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weights = load_safetensors_weights(model_dir)?;

        let embed_tokens =
            MlxEmbedTokens::from_weights(&weights, "model.embed_tokens", qc.group_size, qc.bits);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MlxQuantizedPhi3DecoderLayer::from_weights(
                &weights,
                &format!("model.layers.{i}"),
                config,
                qc,
            )?);
        }

        let mut norm = nn::RmsNormBuilder::new(config.hidden_size as i32)
            .eps(config.rms_norm_eps)
            .build()?;
        crate::models::quantized_llama::assign_weight(
            &mut norm.weight,
            &weights,
            "model.norm.weight",
        );

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

impl super::MlxModel for MlxQuantizedPhi3ForCausalLM {
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

        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };

        logits.as_dtype(Dtype::Float32)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

pub fn create_mlx_phi3(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let phi3_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let model = MlxPhi3ForCausalLM::load(model_dir, &phi3_config, dtype)?;
    Ok(Box::new(model))
}

pub fn create_mlx_quantized_phi3(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let phi3_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let qc = QuantConfig::from_hf_config(config).unwrap_or_default();
    tracing::info!(
        "Loading quantized MLX Phi-3 (group_size={}, bits={})",
        qc.group_size,
        qc.bits
    );
    let model = MlxQuantizedPhi3ForCausalLM::load(model_dir, &phi3_config, &qc, dtype)?;
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
            sliding_window: None,
        }
    }

    #[test]
    fn test_phi3_mlp_forward() {
        let mut mlp = MlxPhi3MLP::new(32, 64).unwrap();
        let x = mlx_rs::ops::ones::<f32>(&[3, 32]).unwrap();
        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[3, 32]);
    }

    #[test]
    fn test_phi3_attention_forward() {
        let config = test_config();
        let mut attn = MlxPhi3Attention::new(&config).unwrap();

        let x = mlx_rs::ops::ones::<f32>(&[4, 32]).unwrap();
        let positions = Array::from_iter(0..4i32, &[4]);
        let mut cache = None;

        let out = attn.forward(&x, &positions, &mut cache).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[4, 32]);
        assert!(cache.is_some());
    }

    #[test]
    fn test_phi3_model_forward() {
        let config = test_config();
        let mut model = MlxPhi3ForCausalLM::new(&config).unwrap();

        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        let logits = <MlxPhi3ForCausalLM as crate::models::MlxModel>::forward(
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
    fn test_phi3_prefill_and_decode() {
        let config = test_config();
        let mut model = MlxPhi3ForCausalLM::new(&config).unwrap();
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        // Prefill
        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let logits = <MlxPhi3ForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
        )
        .unwrap();
        logits.eval().unwrap();

        for entry in &kv_cache {
            assert!(entry.is_some());
        }

        // Decode
        let decode_ids = Array::from_iter(vec![15i32], &[1]);
        let decode_pos = Array::from_iter(vec![3i32], &[1]);
        let logits2 = <MlxPhi3ForCausalLM as crate::models::MlxModel>::forward(
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
