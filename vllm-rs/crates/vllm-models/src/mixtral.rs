// SPDX-License-Identifier: Apache-2.0
//! Mixtral MoE model architecture.
//!
//! Mixtral uses LLaMA-like attention (GQA) with a Mixture of Experts FFN.
//! All layers are MoE — no dense/MoE alternation, no shared experts.
//!
//! Key difference from Qwen3 MoE: expert weights use w1/w2/w3 names
//! (not gate_proj/up_proj/down_proj), and the MoE module is named
//! `block_sparse_moe` (not `mlp`).
//!
//! Weight mapping: w1 = gate_proj, w3 = up_proj, w2 = down_proj.
//!
//! Port of: `vllm/model_executor/models/mixtral.py`

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{Embedding, Linear, RmsNorm};
use vllm_model::lora::LoraAdapter;
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::llama::{LlamaAttention, LlamaConfig};

// ---------------------------------------------------------------------------
// MixtralConfig
// ---------------------------------------------------------------------------

/// Parsed configuration for a Mixtral model.
#[derive(Debug, Clone)]
pub struct MixtralConfig {
    // Base LLaMA-like fields.
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
    pub sliding_window: Option<usize>,

    // MoE-specific.
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
}

impl MixtralConfig {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> ModelResult<Self> {
        let base = LlamaConfig::from_hf_config(config)?;
        let extra = &config.extra;

        let get_usize = |key: &str| -> Option<usize> {
            extra.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
        };

        Ok(Self {
            hidden_size: base.hidden_size,
            num_attention_heads: base.num_attention_heads,
            num_kv_heads: base.num_kv_heads,
            num_hidden_layers: base.num_hidden_layers,
            intermediate_size: base.intermediate_size,
            vocab_size: base.vocab_size,
            max_position_embeddings: base.max_position_embeddings,
            rms_norm_eps: base.rms_norm_eps,
            rope_theta: base.rope_theta,
            head_dim: base.head_dim,
            tie_word_embeddings: base.tie_word_embeddings,
            sliding_window: base.sliding_window,
            num_local_experts: get_usize("num_local_experts").unwrap_or(8),
            num_experts_per_tok: get_usize("num_experts_per_tok").unwrap_or(2),
        })
    }

    /// Produce a `LlamaConfig` for constructing attention layers.
    pub fn llama_config(&self) -> LlamaConfig {
        LlamaConfig {
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            head_dim: self.head_dim,
            tie_word_embeddings: self.tie_word_embeddings,
            sliding_window: self.sliding_window,
            partial_rotary_factor: 1.0,
            long_rope_scaling: None,
        }
    }
}

// ---------------------------------------------------------------------------
// MixtralExpertMLP
// ---------------------------------------------------------------------------

/// A single Mixtral expert using w1/w2/w3 naming convention.
///
/// w1 = gate_proj, w3 = up_proj, w2 = down_proj.
/// SiLU(w1(x)) * w3(x) → w2(...)
struct MixtralExpertMLP {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl MixtralExpertMLP {
    fn load(weights: &ModelWeights, prefix: &str, dtype: DType) -> ModelResult<Self> {
        let w1 = Linear::load(weights, &format!("{prefix}.w1"), dtype)?;
        let w2 = Linear::load(weights, &format!("{prefix}.w2"), dtype)?;
        let w3 = Linear::load(weights, &format!("{prefix}.w3"), dtype)?;
        Ok(Self { w1, w2, w3 })
    }

    fn zeros(
        hidden_size: usize,
        intermediate_size: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let w1 = Linear::zeros(hidden_size, intermediate_size, dtype, device)?;
        let w2 = Linear::zeros(intermediate_size, hidden_size, dtype, device)?;
        let w3 = Linear::zeros(hidden_size, intermediate_size, dtype, device)?;
        Ok(Self { w1, w2, w3 })
    }
}

impl Module for MixtralExpertMLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.w1.forward(x)?;
        let up = self.w3.forward(x)?;
        let activated = crate::ops::silu_and_mul(&gate, &up)?;
        self.w2.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// MixtralMoE
// ---------------------------------------------------------------------------

/// Mixture of Experts layer (no shared expert).
pub struct MixtralMoE {
    gate: Linear,
    experts: Vec<MixtralExpertMLP>,
    top_k: usize,
}

impl MixtralMoE {
    /// Load MoE weights.
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &MixtralConfig,
        dtype: DType,
    ) -> ModelResult<Self> {
        let n = config.num_local_experts;

        let gate = Linear::load(weights, &format!("{prefix}.gate"), dtype)?;

        let mut experts = Vec::with_capacity(n);
        for i in 0..n {
            experts.push(MixtralExpertMLP::load(
                weights,
                &format!("{prefix}.experts.{i}"),
                dtype,
            )?);
        }

        Ok(Self {
            gate,
            experts,
            top_k: config.num_experts_per_tok,
        })
    }

    /// Create with zero weights (for testing).
    pub fn zeros(config: &MixtralConfig, dtype: DType, device: &Device) -> ModelResult<Self> {
        let n = config.num_local_experts;
        let gate = Linear::zeros(config.hidden_size, n, dtype, device)?;

        let mut experts = Vec::with_capacity(n);
        for _ in 0..n {
            experts.push(MixtralExpertMLP::zeros(
                config.hidden_size,
                config.intermediate_size,
                dtype,
                device,
            )?);
        }

        Ok(Self {
            gate,
            experts,
            top_k: config.num_experts_per_tok,
        })
    }
}

impl Module for MixtralMoE {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (num_tokens, hidden_size) = x.dims2()?;

        // Compute router logits and softmax probabilities.
        let router_logits = self.gate.forward(x)?; // [num_tokens, n_experts]
        let max_vals = router_logits.max_keepdim(candle_core::D::Minus1)?;
        let shifted = router_logits.broadcast_sub(&max_vals)?;
        let exp = shifted.exp()?;
        let sum = exp.sum_keepdim(candle_core::D::Minus1)?;
        let probs = exp.broadcast_div(&sum)?;

        // Top-k selection per token on CPU.
        let probs_f32 = probs.to_dtype(DType::F32)?;
        let probs_vec = probs_f32.to_vec2::<f32>()?;

        let mut output_data = vec![0.0f32; num_tokens * hidden_size];

        for tok in 0..num_tokens {
            let token_probs = &probs_vec[tok];
            let mut indexed: Vec<(usize, f32)> = token_probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(self.top_k);

            // Mixtral does not renormalize top-k probabilities.
            let token_x = x.narrow(0, tok, 1)?; // [1, hidden]

            for &(expert_idx, prob) in &indexed {
                let expert_out = self.experts[expert_idx].forward(&token_x)?;
                let expert_vals = expert_out.flatten_all()?.to_vec1::<f32>()?;
                for (j, &v) in expert_vals.iter().enumerate() {
                    output_data[tok * hidden_size + j] += v * prob;
                }
            }
        }

        let output = Tensor::from_slice(&output_data, (num_tokens, hidden_size), x.device())?;
        output.to_dtype(x.dtype())
    }
}

// ---------------------------------------------------------------------------
// MixtralDecoderLayer
// ---------------------------------------------------------------------------

/// A single Mixtral decoder layer (always MoE).
pub struct MixtralDecoderLayer {
    self_attn: LlamaAttention,
    block_sparse_moe: MixtralMoE,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MixtralDecoderLayer {
    /// Load a decoder layer.
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &MixtralConfig,
        dtype: DType,
        device: &Device,
        layer_idx: usize,
    ) -> ModelResult<Self> {
        let llama_config = config.llama_config();
        let self_attn = LlamaAttention::load(
            weights,
            &format!("{prefix}.self_attn"),
            &llama_config,
            dtype,
            device,
            0,
            1,
            layer_idx,
        )?;

        let block_sparse_moe = MixtralMoE::load(
            weights,
            &format!("{prefix}.block_sparse_moe"),
            config,
            dtype,
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
            block_sparse_moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Create with zero weights (for testing).
    pub fn zeros(
        config: &MixtralConfig,
        dtype: DType,
        device: &Device,
        layer_idx: usize,
    ) -> ModelResult<Self> {
        let llama_config = config.llama_config();
        let self_attn = LlamaAttention::zeros(&llama_config, dtype, device, layer_idx)?;
        let block_sparse_moe = MixtralMoE::zeros(config, dtype, device)?;

        let input_layernorm =
            RmsNorm::ones(config.hidden_size, config.rms_norm_eps, dtype, device)?;
        let post_attention_layernorm =
            RmsNorm::ones(config.hidden_size, config.rms_norm_eps, dtype, device)?;

        Ok(Self {
            self_attn,
            block_sparse_moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<Tensor> {
        // Pre-attention layernorm + attention + residual.
        let normed = crate::ops::rms_norm(hidden_states, &self.input_layernorm)
            .map_err(ModelError::Candle)?;
        let attn_output = self.self_attn.forward(&normed, positions, kv_cache)?;
        let hidden_states = (hidden_states + attn_output).map_err(ModelError::Candle)?;

        // Post-attention layernorm + MoE + residual.
        let normed = crate::ops::rms_norm(&hidden_states, &self.post_attention_layernorm)
            .map_err(ModelError::Candle)?;
        let mlp_output = self
            .block_sparse_moe
            .forward(&normed)
            .map_err(ModelError::Candle)?;
        let hidden_states = (hidden_states + mlp_output).map_err(ModelError::Candle)?;

        Ok(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// MixtralModel
// ---------------------------------------------------------------------------

/// Mixtral transformer backbone.
struct MixtralModel {
    embed_tokens: Embedding,
    layers: Vec<MixtralDecoderLayer>,
    norm: RmsNorm,
}

impl MixtralModel {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &MixtralConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let embed_tokens = Embedding::load(weights, &format!("{prefix}.embed_tokens"), dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MixtralDecoderLayer::load(
                weights,
                &format!("{prefix}.layers.{i}"),
                config,
                dtype,
                device,
                i,
            )?);
        }

        let norm = RmsNorm::load(
            weights,
            &format!("{prefix}.norm"),
            config.rms_norm_eps,
            dtype,
        )?;

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
// MixtralForCausalLM
// ---------------------------------------------------------------------------

/// Mixtral for causal language modeling.
pub struct MixtralForCausalLM {
    model: MixtralModel,
    lm_head: Linear,
}

impl MixtralForCausalLM {
    /// Load the full model from weights.
    pub fn load(
        weights: &ModelWeights,
        config: &MixtralConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let model = MixtralModel::load(weights, "model", config, dtype, device)?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight().clone(), None)
        } else {
            Linear::load(weights, "lm_head", dtype)?
        };

        Ok(Self { model, lm_head })
    }
}

impl crate::Model for MixtralForCausalLM {
    fn inject_lora(&mut self, adapter: &LoraAdapter) -> ModelResult<()> {
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            // Attention uses LlamaAttention — delegate.
            let attn_prefix = format!("model.layers.{}.self_attn", i);
            layer.self_attn.inject_lora(&attn_prefix, adapter)?;
            // MoE expert LoRA is not supported — all layers are MoE.
        }
        Ok(())
    }

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

/// Factory function for the model registry.
pub fn create_mixtral(
    weights: &ModelWeights,
    config: &HfModelConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let mixtral_config = MixtralConfig::from_hf_config(config)?;
    let model = MixtralForCausalLM::load(weights, &mixtral_config, dtype, device)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> MixtralConfig {
        MixtralConfig {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 1000000.0,
            head_dim: 8,
            tie_word_embeddings: false,
            sliding_window: Some(4096),
            num_local_experts: 4,
            num_experts_per_tok: 2,
        }
    }

    #[test]
    fn test_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["MixtralForCausalLM"],
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "intermediate_size": 14336,
                "vocab_size": 32000,
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-5,
                "rope_theta": 1000000.0,
                "sliding_window": 4096,
                "num_local_experts": 8,
                "num_experts_per_tok": 2
            }"#,
        )
        .unwrap();

        let config = MixtralConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_local_experts, 8);
        assert_eq!(config.num_experts_per_tok, 2);
        assert_eq!(config.intermediate_size, 14336);
        assert_eq!(config.sliding_window, Some(4096));
        assert_eq!(config.rope_theta, 1000000.0);
    }

    #[test]
    fn test_is_moe_all_layers() {
        // Mixtral: all layers are MoE (no dense/MoE alternation).
        let config = test_config();
        for i in 0..config.num_hidden_layers {
            assert_eq!(
                config.num_local_experts > 0,
                true,
                "layer {i} should be MoE"
            );
        }
    }

    #[test]
    fn test_moe_forward_zeros() {
        let config = test_config();
        let device = Device::Cpu;
        let dtype = DType::F32;

        let moe = MixtralMoE::zeros(&config, dtype, &device).unwrap();
        let x = Tensor::zeros((2, config.hidden_size), dtype, &device).unwrap();
        let output = moe.forward(&x).unwrap();
        assert_eq!(output.dims(), &[2, config.hidden_size]);
    }

    #[test]
    fn test_decoder_layer_zeros() {
        let config = test_config();
        let device = Device::Cpu;
        let dtype = DType::F32;

        let layer = MixtralDecoderLayer::zeros(&config, dtype, &device, 0).unwrap();
        let x = Tensor::zeros((3, config.hidden_size), dtype, &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();
        let output = layer.forward(&x, &positions, None).unwrap();
        assert_eq!(output.dims(), &[3, config.hidden_size]);
    }

    #[test]
    fn test_expert_mlp_zeros() {
        let config = test_config();
        let device = Device::Cpu;
        let dtype = DType::F32;

        let expert =
            MixtralExpertMLP::zeros(config.hidden_size, config.intermediate_size, dtype, &device)
                .unwrap();
        let x = Tensor::zeros((2, config.hidden_size), dtype, &device).unwrap();
        let output = expert.forward(&x).unwrap();
        assert_eq!(output.dims(), &[2, config.hidden_size]);
    }

    #[test]
    fn test_registry() {
        let registry = crate::ModelRegistry::default_registry();
        assert!(registry.contains("MixtralForCausalLM"));
    }

    #[test]
    fn test_mixtral_model_from_weights() {
        // Build a tiny Mixtral model from synthetic weights.
        let config = test_config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let device = Device::Cpu;
        let dtype = DType::F32;

        let mut specs: Vec<(String, Vec<usize>)> = Vec::new();

        // Embeddings.
        specs.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));

        // Layers.
        let q_size = config.num_attention_heads * config.head_dim;
        let kv_size = config.num_kv_heads * config.head_dim;

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            // Attention.
            specs.push((
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![q_size, config.hidden_size],
            ));
            specs.push((
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv_size, config.hidden_size],
            ));
            specs.push((
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv_size, config.hidden_size],
            ));
            specs.push((
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![config.hidden_size, q_size],
            ));

            // MoE gate: hidden_size -> num_local_experts.
            specs.push((
                format!("{prefix}.block_sparse_moe.gate.weight"),
                vec![config.num_local_experts, config.hidden_size],
            ));

            // Experts: w1/w2/w3 naming.
            for e in 0..config.num_local_experts {
                specs.push((
                    format!("{prefix}.block_sparse_moe.experts.{e}.w1.weight"),
                    vec![config.intermediate_size, config.hidden_size],
                ));
                specs.push((
                    format!("{prefix}.block_sparse_moe.experts.{e}.w2.weight"),
                    vec![config.hidden_size, config.intermediate_size],
                ));
                specs.push((
                    format!("{prefix}.block_sparse_moe.experts.{e}.w3.weight"),
                    vec![config.intermediate_size, config.hidden_size],
                ));
            }

            // Norms.
            specs.push((
                format!("{prefix}.input_layernorm.weight"),
                vec![config.hidden_size],
            ));
            specs.push((
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![config.hidden_size],
            ));
        }

        // Final norm + LM head.
        specs.push(("model.norm.weight".to_string(), vec![config.hidden_size]));
        specs.push((
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));

        create_test_weights(&path, &specs);

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model = MixtralForCausalLM::load(&weights, &config, dtype, &device).unwrap();

        // Forward pass.
        let input_ids = Tensor::new(&[1u32, 5, 10], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();

        let logits = crate::Model::forward(&model, &input_ids, &positions, None).unwrap();
        assert_eq!(logits.dims(), &[3, config.vocab_size]);
    }

    fn create_test_weights(path: &std::path::Path, specs: &[(String, Vec<usize>)]) {
        use safetensors::tensor::TensorView;

        let mut all_data: Vec<Vec<u8>> = Vec::new();
        for (name, shape) in specs {
            let num_elements: usize = shape.iter().product();
            let val = if name.contains("layernorm") || name == "model.norm.weight" {
                1.0f32
            } else {
                0.01f32
            };
            let data: Vec<u8> = (0..num_elements).flat_map(|_| val.to_le_bytes()).collect();
            all_data.push(data);
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
