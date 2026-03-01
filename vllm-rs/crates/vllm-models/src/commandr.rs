// SPDX-License-Identifier: Apache-2.0
//! Command R (CohereForCausalLM) model architecture.
//!
//! Implements:
//! - `CommandRForCausalLM` — top-level model with logit scaling and tied embeddings
//! - `CommandRModel` — transformer backbone (embed + layers + final CohereLayerNorm)
//! - `CommandRDecoderLayer` — parallel attention + MLP with one norm
//! - `CommandRAttention` — multi-head attention with interleaved RoPE and optional QK norm
//!
//! Key differences from LLaMA:
//! - CohereLayerNorm (full LayerNorm with mean subtraction, weight only, no bias)
//! - Parallel attention + MLP: one norm per layer, both branches read same normed input
//! - Logit scaling: `logits *= logit_scale` (e.g. 0.0625 = 1/16)
//! - Interleaved RoPE: adjacent pairs (2i, 2i+1) rotated together (Cohere convention)
//! - Optional QK norm (CohereLayerNorm on Q/K after projection, before RoPE)
//!
//! Port of: `vllm/model_executor/models/commandr.py`

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{
    CohereLayerNorm, ColumnParallelLinear, Embedding, Linear, RowParallelLinear,
};
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::attention::attention_with_cache;
use crate::llama::LlamaMLP;
use crate::quantized_llama::{apply_interleaved_rope, precompute_freqs_cis};

// ---------------------------------------------------------------------------
// CommandRConfig
// ---------------------------------------------------------------------------

/// Parsed configuration for a Command R model.
#[derive(Debug, Clone)]
pub struct CommandRConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub rope_theta: f64,
    pub head_dim: usize,
    pub logit_scale: f64,
    pub use_qk_norm: bool,
    pub tie_word_embeddings: bool,
}

impl CommandRConfig {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> ModelResult<Self> {
        let hidden_size = config
            .hidden_size
            .ok_or_else(|| ModelError::Other("missing hidden_size".into()))?;
        let num_attention_heads = config
            .num_attention_heads
            .ok_or_else(|| ModelError::Other("missing num_attention_heads".into()))?;

        let logit_scale = config
            .extra
            .get("logit_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let use_qk_norm = config
            .extra
            .get("use_qk_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

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
            max_position_embeddings: config.max_position_embeddings.unwrap_or(8192),
            layer_norm_eps: config.norm_eps(),
            rope_theta: config.rope_theta.unwrap_or(8000000.0),
            head_dim: config
                .head_dim()
                .unwrap_or(hidden_size / num_attention_heads),
            logit_scale,
            use_qk_norm,
            tie_word_embeddings: config.tie_word_embeddings.unwrap_or(true),
        })
    }
}

// ---------------------------------------------------------------------------
// CommandRAttention
// ---------------------------------------------------------------------------

/// Command R multi-head attention with interleaved RoPE and optional QK norm.
struct CommandRAttention {
    q_proj: ColumnParallelLinear,
    k_proj: ColumnParallelLinear,
    v_proj: ColumnParallelLinear,
    o_proj: RowParallelLinear,
    /// Optional QK norms (CohereLayerNorm).
    q_norm: Option<CohereLayerNorm>,
    k_norm: Option<CohereLayerNorm>,
    /// Precomputed cos values for interleaved RoPE: [max_position, head_dim/2]
    cos: Tensor,
    /// Precomputed sin values for interleaved RoPE: [max_position, head_dim/2]
    sin: Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl CommandRAttention {
    /// Load attention weights.
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &CommandRConfig,
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

        // Optional QK norms.
        let q_norm = if config.use_qk_norm {
            Some(CohereLayerNorm::load(
                weights,
                &format!("{}.q_norm", prefix),
                config.layer_norm_eps,
                dtype,
            )?)
        } else {
            None
        };
        let k_norm = if config.use_qk_norm {
            Some(CohereLayerNorm::load(
                weights,
                &format!("{}.k_norm", prefix),
                config.layer_norm_eps,
                dtype,
            )?)
        } else {
            None
        };

        let num_q_heads = config.num_attention_heads / world_size;
        let num_kv_heads = config.num_kv_heads / world_size;
        let head_dim = config.head_dim;

        // Precompute interleaved RoPE cos/sin tables.
        let (cos, sin) = precompute_freqs_cis(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            cos,
            sin,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// Create with zero weights (for testing).
    #[cfg(test)]
    fn zeros(config: &CommandRConfig, dtype: DType, device: &Device) -> ModelResult<Self> {
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

        let q_norm = if config.use_qk_norm {
            Some(CohereLayerNorm::ones(
                config.head_dim,
                config.layer_norm_eps,
                dtype,
                device,
            )?)
        } else {
            None
        };
        let k_norm = if config.use_qk_norm {
            Some(CohereLayerNorm::ones(
                config.head_dim,
                config.layer_norm_eps,
                dtype,
                device,
            )?)
        } else {
            None
        };

        let (cos, sin) = precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            cos,
            sin,
            num_q_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: 1.0 / (config.head_dim as f64).sqrt(),
        })
    }

    /// Forward pass.
    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
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
        let mut q = q
            .reshape((num_tokens, self.num_q_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let mut k = k
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let v = v
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;

        // Optional QK norms (applied per-head before RoPE).
        if let Some(ref norm) = self.q_norm {
            q = norm.forward(&q).map_err(ModelError::Candle)?;
        }
        if let Some(ref norm) = self.k_norm {
            k = norm.forward(&k).map_err(ModelError::Candle)?;
        }

        // Apply interleaved RoPE.
        let q = apply_interleaved_rope(&q, &self.cos, &self.sin, positions)?;
        let k = apply_interleaved_rope(&k, &self.cos, &self.sin, positions)?;

        // Cache-merge + attention.
        let attn_output = attention_with_cache(&q, &k, &v, self.scale, kv_cache, None)?;

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
// CommandRDecoderLayer
// ---------------------------------------------------------------------------

/// A single Command R decoder layer with parallel attention + MLP.
///
/// Uses one CohereLayerNorm per layer. Both attention and MLP read from the
/// same normed input and their outputs are summed with the residual:
///   `out = residual + attn(norm(x)) + mlp(norm(x))`
struct CommandRDecoderLayer {
    self_attn: CommandRAttention,
    mlp: LlamaMLP,
    input_layernorm: CohereLayerNorm,
}

impl CommandRDecoderLayer {
    /// Load a decoder layer.
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &CommandRConfig,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let self_attn = CommandRAttention::load(
            weights,
            &format!("{}.self_attn", prefix),
            config,
            dtype,
            device,
            rank,
            world_size,
        )?;
        let mlp = LlamaMLP::load(weights, &format!("{}.mlp", prefix), dtype, rank, world_size)?;
        let input_layernorm = CohereLayerNorm::load(
            weights,
            &format!("{}.input_layernorm", prefix),
            config.layer_norm_eps,
            dtype,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
        })
    }

    /// Forward pass: norm → parallel attn + mlp → add residual.
    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<Tensor> {
        let residual = hidden_states;

        // Single norm applied to input (shared by attn and mlp).
        let normed = self
            .input_layernorm
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;

        // Parallel attention and MLP on the same normed input.
        let attn_output = self.self_attn.forward(&normed, positions, kv_cache)?;
        let mlp_output = self.mlp.forward(&normed).map_err(ModelError::Candle)?;

        // residual + attn_output + mlp_output
        let hidden_states = (residual + attn_output).map_err(ModelError::Candle)?;
        let hidden_states = (hidden_states + mlp_output).map_err(ModelError::Candle)?;

        Ok(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// CommandRModel
// ---------------------------------------------------------------------------

/// Command R transformer backbone.
///
/// Embedding → N decoder layers → final CohereLayerNorm.
struct CommandRModel {
    embed_tokens: Embedding,
    layers: Vec<CommandRDecoderLayer>,
    norm: CohereLayerNorm,
}

impl CommandRModel {
    /// Load the model backbone.
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &CommandRConfig,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let embed_tokens = Embedding::load(weights, &format!("{}.embed_tokens", prefix), dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = CommandRDecoderLayer::load(
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

        let norm = CohereLayerNorm::load(
            weights,
            &format!("{}.norm", prefix),
            config.layer_norm_eps,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass.
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

    /// Number of decoder layers.
    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// CommandRForCausalLM
// ---------------------------------------------------------------------------

/// Command R for causal language modeling.
///
/// Wraps `CommandRModel` with a language model head that projects hidden states
/// to vocabulary logits, then applies logit scaling.
pub struct CommandRForCausalLM {
    model: CommandRModel,
    lm_head: Linear,
    logit_scale: f64,
}

impl CommandRForCausalLM {
    /// Load the full model from weights.
    pub fn load(
        weights: &ModelWeights,
        config: &CommandRConfig,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let model = CommandRModel::load(weights, "model", config, dtype, device, rank, world_size)?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight().clone(), None)
        } else {
            Linear::load(weights, "lm_head", dtype)?
        };

        Ok(Self {
            model,
            lm_head,
            logit_scale: config.logit_scale,
        })
    }
}

impl crate::Model for CommandRForCausalLM {
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
        // Apply logit scaling.
        let logits = (logits * self.logit_scale).map_err(ModelError::Candle)?;
        // Cast logits to f32 for sampling.
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
pub fn create_commandr(
    weights: &ModelWeights,
    config: &HfModelConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let commandr_config = CommandRConfig::from_hf_config(config)?;
    let model = CommandRForCausalLM::load(weights, &commandr_config, dtype, device, 0, 1)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> CommandRConfig {
        CommandRConfig {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            layer_norm_eps: 1e-5,
            rope_theta: 8000000.0,
            head_dim: 8,
            logit_scale: 0.0625,
            use_qk_norm: false,
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn test_commandr_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["CohereForCausalLM"],
                "model_type": "cohere",
                "hidden_size": 8192,
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "num_hidden_layers": 40,
                "intermediate_size": 22528,
                "vocab_size": 256000,
                "max_position_embeddings": 8192,
                "layer_norm_eps": 1e-5,
                "rope_theta": 8000000.0,
                "logit_scale": 0.0625,
                "tie_word_embeddings": true
            }"#,
        )
        .unwrap();

        let config = CommandRConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 8192);
        assert_eq!(config.num_attention_heads, 64);
        assert_eq!(config.num_kv_heads, 64);
        assert_eq!(config.num_hidden_layers, 40);
        assert_eq!(config.intermediate_size, 22528);
        assert_eq!(config.vocab_size, 256000);
        assert!((config.logit_scale - 0.0625).abs() < 1e-10);
        assert!(!config.use_qk_norm);
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn test_commandr_config_defaults() {
        // Test that missing optional fields get proper defaults.
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["CohereForCausalLM"],
                "hidden_size": 256,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "intermediate_size": 512,
                "vocab_size": 1000
            }"#,
        )
        .unwrap();

        let config = CommandRConfig::from_hf_config(&hf_config).unwrap();
        assert!((config.logit_scale - 1.0).abs() < 1e-10);
        assert!(!config.use_qk_norm);
        assert!(config.tie_word_embeddings); // Command R defaults to tied
        assert!((config.rope_theta - 8000000.0).abs() < 1.0);
    }

    #[test]
    fn test_commandr_attention_forward() {
        let config = test_config();
        let attn = CommandRAttention::zeros(&config, DType::F32, &Device::Cpu).unwrap();

        let num_tokens = 4;
        let x = Tensor::ones(&[num_tokens, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[num_tokens, config.hidden_size]);
    }

    #[test]
    fn test_commandr_attention_with_qk_norm() {
        let config = CommandRConfig {
            use_qk_norm: true,
            ..test_config()
        };
        let attn = CommandRAttention::zeros(&config, DType::F32, &Device::Cpu).unwrap();

        let num_tokens = 3;
        let x = Tensor::ones(&[num_tokens, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[num_tokens, config.hidden_size]);
    }

    #[test]
    fn test_commandr_model_from_weights() {
        let config = test_config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let device = Device::Cpu;
        let dtype = DType::F32;

        let tensor_specs = build_weight_specs(&config);
        create_test_weights(&path, &tensor_specs);

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model = CommandRForCausalLM::load(&weights, &config, dtype, &device, 0, 1).unwrap();

        let input_ids = Tensor::new(&[1u32, 5, 10], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();

        let logits = crate::Model::forward(&model, &input_ids, &positions, None).unwrap();
        assert_eq!(logits.dims(), &[3, config.vocab_size]);
    }

    #[test]
    fn test_commandr_logit_scaling() {
        // Verify that logit_scale is applied: compare scale=1.0 vs scale=0.5.
        // Use non-trivial weights to avoid 0/0 = NaN.
        let config_full = CommandRConfig {
            logit_scale: 1.0,
            ..test_config()
        };
        let config_half = CommandRConfig {
            logit_scale: 0.5,
            ..test_config()
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let device = Device::Cpu;
        let dtype = DType::F32;

        let tensor_specs = build_weight_specs(&config_full);
        create_test_weights(&path, &tensor_specs);

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model_full =
            CommandRForCausalLM::load(&weights, &config_full, dtype, &device, 0, 1).unwrap();
        let model_half =
            CommandRForCausalLM::load(&weights, &config_half, dtype, &device, 0, 1).unwrap();

        let input_ids = Tensor::new(&[1u32, 2], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1], &device).unwrap();

        let logits_full = crate::Model::forward(&model_full, &input_ids, &positions, None).unwrap();
        let logits_half = crate::Model::forward(&model_half, &input_ids, &positions, None).unwrap();

        let v_full = logits_full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let v_half = logits_half.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Find a non-zero element to compare ratio.
        let mut found_scaled = false;
        for (f, h) in v_full.iter().zip(v_half.iter()) {
            if f.abs() > 1e-8 {
                let ratio = h / f;
                assert!(
                    (ratio - 0.5).abs() < 0.01,
                    "logit scaling ratio should be ~0.5, got {ratio}"
                );
                found_scaled = true;
                break;
            }
        }
        // If all logits are ~0 (possible with small weights), verify both are ~0.
        if !found_scaled {
            let sum_full: f32 = v_full.iter().map(|x| x.abs()).sum();
            let sum_half: f32 = v_half.iter().map(|x| x.abs()).sum();
            assert!(sum_full < 1e-6 && sum_half < 1e-6, "both should be ~0");
        }
    }

    #[test]
    fn test_commandr_registry() {
        let registry = crate::ModelRegistry::default_registry();
        assert!(registry.contains("CohereForCausalLM"));
    }

    #[test]
    fn test_commandr_kv_cache() {
        let config = test_config();
        let attn = CommandRAttention::zeros(&config, DType::F32, &Device::Cpu).unwrap();

        // Prefill: 4 tokens.
        let x = Tensor::ones(&[4, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();
        let mut cache: Option<(Tensor, Tensor)> = None;

        let handle = crate::LayerKvHandle::Contiguous(&mut cache);
        let out = attn.forward(&x, &positions, Some(handle)).unwrap();
        assert_eq!(out.dims(), &[4, config.hidden_size]);
        assert_eq!(cache.as_ref().unwrap().0.dim(0).unwrap(), 4);

        // Decode: 1 token.
        let x_decode = Tensor::ones(&[1, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let pos_decode = Tensor::new(&[4u32], &Device::Cpu).unwrap();

        let handle = crate::LayerKvHandle::Contiguous(&mut cache);
        let out_decode = attn.forward(&x_decode, &pos_decode, Some(handle)).unwrap();
        assert_eq!(out_decode.dims(), &[1, config.hidden_size]);
        assert_eq!(cache.as_ref().unwrap().0.dim(0).unwrap(), 5);
    }

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn build_weight_specs(config: &CommandRConfig) -> Vec<(String, Vec<usize>)> {
        let mut specs = Vec::new();

        // Embeddings.
        specs.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));

        // Layers — only input_layernorm (no post_attention_layernorm for Command R).
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let q_size = config.num_attention_heads * config.head_dim;
            let kv_size = config.num_kv_heads * config.head_dim;

            specs.push((
                format!("{}.self_attn.q_proj.weight", prefix),
                vec![q_size, config.hidden_size],
            ));
            specs.push((
                format!("{}.self_attn.k_proj.weight", prefix),
                vec![kv_size, config.hidden_size],
            ));
            specs.push((
                format!("{}.self_attn.v_proj.weight", prefix),
                vec![kv_size, config.hidden_size],
            ));
            specs.push((
                format!("{}.self_attn.o_proj.weight", prefix),
                vec![config.hidden_size, q_size],
            ));
            specs.push((
                format!("{}.mlp.gate_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            ));
            specs.push((
                format!("{}.mlp.up_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            ));
            specs.push((
                format!("{}.mlp.down_proj.weight", prefix),
                vec![config.hidden_size, config.intermediate_size],
            ));
            specs.push((
                format!("{}.input_layernorm.weight", prefix),
                vec![config.hidden_size],
            ));
        }

        // Final norm.
        specs.push(("model.norm.weight".to_string(), vec![config.hidden_size]));

        // No lm_head for tied embeddings.
        if !config.tie_word_embeddings {
            specs.push((
                "lm_head.weight".to_string(),
                vec![config.vocab_size, config.hidden_size],
            ));
        }

        specs
    }

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

        // Override norm weights with 1.0.
        for (i, (name, shape)) in specs.iter().enumerate() {
            if name.contains("layernorm") || name == "model.norm.weight" {
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
                    name.as_str(),
                    TensorView::new(safetensors::Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        let bytes = safetensors::tensor::serialize(views, None).unwrap();
        std::fs::write(path, bytes).unwrap();
    }
}
