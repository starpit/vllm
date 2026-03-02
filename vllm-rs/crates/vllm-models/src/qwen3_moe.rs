// SPDX-License-Identifier: Apache-2.0
//! Qwen3 MoE / Qwen2 MoE model architecture.
//!
//! Both Qwen3 MoE and Qwen2 MoE share the same architecture:
//! - LLaMA-like attention (Qwen3 adds QK norms, Qwen2 has QKV bias — both
//!   handled by the layer loading automatically)
//! - MoE routing: gate -> softmax -> top-k + sigmoid-gated shared expert
//! - Dense/MoE layer selection via `decoder_sparse_step` + `mlp_only_layers`
//!
//! Port of: `vllm/model_executor/models/qwen3_moe.py`

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{Embedding, Linear, RmsNorm};
use vllm_model::lora::LoraAdapter;
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::llama::{LlamaAttention, LlamaConfig, LlamaMLP};

// ---------------------------------------------------------------------------
// Qwen3MoeConfig
// ---------------------------------------------------------------------------

/// Parsed configuration for a Qwen3 MoE / Qwen2 MoE model.
#[derive(Debug, Clone)]
pub struct Qwen3MoeConfig {
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

    // MoE-specific.
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub decoder_sparse_step: usize,
    pub mlp_only_layers: Vec<usize>,
}

impl Qwen3MoeConfig {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> ModelResult<Self> {
        let base = LlamaConfig::from_hf_config(config)?;
        let extra = &config.extra;

        let get_usize = |key: &str| -> Option<usize> {
            extra.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
        };
        let get_bool = |key: &str| -> Option<bool> { extra.get(key).and_then(|v| v.as_bool()) };

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
            num_experts: get_usize("num_experts").unwrap_or(0),
            num_experts_per_tok: get_usize("num_experts_per_tok").unwrap_or(4),
            moe_intermediate_size: get_usize("moe_intermediate_size")
                .unwrap_or(base.intermediate_size),
            shared_expert_intermediate_size: get_usize("shared_expert_intermediate_size")
                .unwrap_or(0),
            norm_topk_prob: get_bool("norm_topk_prob").unwrap_or(true),
            decoder_sparse_step: get_usize("decoder_sparse_step").unwrap_or(1),
            mlp_only_layers: extra
                .get("mlp_only_layers")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default(),
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
            sliding_window: None,
            partial_rotary_factor: 1.0,
            long_rope_scaling: None,
        }
    }

    /// Whether a given layer index is a MoE layer.
    ///
    /// From Python qwen3_moe.py:
    /// ```python
    /// if (layer_idx not in mlp_only_layers) and \
    ///    (config.num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0):
    ///     # MoE layer
    /// ```
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }
}

// ---------------------------------------------------------------------------
// Sigmoid helper
// ---------------------------------------------------------------------------

/// Element-wise sigmoid: 1 / (1 + exp(-x)).
fn tensor_sigmoid(x: &Tensor) -> candle_core::Result<Tensor> {
    (x.neg()?.exp()? + 1.0)?.recip()
}

// ---------------------------------------------------------------------------
// Qwen3MoE
// ---------------------------------------------------------------------------

/// Mixture of Experts layer with sigmoid-gated shared expert.
///
/// Routes each token to the top-k experts via a gating network, then adds
/// a shared expert contribution gated by `sigmoid(shared_expert_gate(x))`.
pub struct Qwen3MoE {
    gate: Linear,
    experts: Vec<LlamaMLP>,
    shared_expert: Option<LlamaMLP>,
    shared_expert_gate: Option<Linear>,
    top_k: usize,
    norm_topk_prob: bool,
}

impl Qwen3MoE {
    /// Load MoE weights.
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Qwen3MoeConfig,
        dtype: DType,
    ) -> ModelResult<Self> {
        let n = config.num_experts;

        // Gate: hidden_size -> num_experts (no bias).
        let gate = Linear::load(weights, &format!("{prefix}.gate"), dtype)?;

        // Individual experts.
        let mut experts = Vec::with_capacity(n);
        for i in 0..n {
            experts.push(LlamaMLP::load(
                weights,
                &format!("{prefix}.experts.{i}"),
                dtype,
                0,
                1,
            )?);
        }

        // Shared expert + sigmoid gate.
        let (shared_expert, shared_expert_gate) = if config.shared_expert_intermediate_size > 0 {
            let se = LlamaMLP::load(weights, &format!("{prefix}.shared_expert"), dtype, 0, 1)?;
            let seg = Linear::load(weights, &format!("{prefix}.shared_expert_gate"), dtype)?;
            (Some(se), Some(seg))
        } else {
            (None, None)
        };

        Ok(Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
        })
    }

    /// Create with zero weights (for testing).
    pub fn zeros(config: &Qwen3MoeConfig, dtype: DType, device: &Device) -> ModelResult<Self> {
        let n = config.num_experts;
        let gate = Linear::zeros(config.hidden_size, n, dtype, device)?;

        let mut experts = Vec::with_capacity(n);
        for _ in 0..n {
            experts.push(LlamaMLP::zeros(
                config.hidden_size,
                config.moe_intermediate_size,
                dtype,
                device,
            )?);
        }

        let (shared_expert, shared_expert_gate) = if config.shared_expert_intermediate_size > 0 {
            let se = LlamaMLP::zeros(
                config.hidden_size,
                config.shared_expert_intermediate_size,
                dtype,
                device,
            )?;
            let seg = Linear::zeros(config.hidden_size, 1, dtype, device)?;
            (Some(se), Some(seg))
        } else {
            (None, None)
        };

        Ok(Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
        })
    }
}

impl Module for Qwen3MoE {
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

            let total: f32 = indexed.iter().map(|(_, p)| p).sum();
            let scale = if self.norm_topk_prob && total > 0.0 {
                1.0 / total
            } else {
                1.0
            };

            let token_x = x.narrow(0, tok, 1)?; // [1, hidden]

            for &(expert_idx, prob) in &indexed {
                let expert_out = self.experts[expert_idx].forward(&token_x)?;
                let weight = prob * scale;
                let expert_vals = expert_out.flatten_all()?.to_vec1::<f32>()?;
                for (j, &v) in expert_vals.iter().enumerate() {
                    output_data[tok * hidden_size + j] += v * weight;
                }
            }
        }

        let mut output = Tensor::from_slice(&output_data, (num_tokens, hidden_size), x.device())?;
        output = output.to_dtype(x.dtype())?;

        // Shared expert with sigmoid gate.
        if let (Some(shared), Some(seg)) = (&self.shared_expert, &self.shared_expert_gate) {
            let shared_out = shared.forward(x)?;
            let gate_val = seg.forward(x)?; // [num_tokens, 1]
            let gate_sigmoid = tensor_sigmoid(&gate_val)?;
            let shared_gated = shared_out.broadcast_mul(&gate_sigmoid)?;
            output = (output + shared_gated)?;
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Qwen3MoeDecoderLayer
// ---------------------------------------------------------------------------

/// A single Qwen3 MoE decoder layer.
///
/// Dense layers use standard `LlamaMLP`, MoE layers use `Qwen3MoE`.
pub struct Qwen3MoeDecoderLayer {
    self_attn: LlamaAttention,
    mlp: Qwen3MoeMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

/// Either a dense MLP or a MoE layer.
enum Qwen3MoeMlp {
    Dense(LlamaMLP),
    MoE(Qwen3MoE),
}

impl Module for Qwen3MoeMlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Qwen3MoeMlp::Dense(mlp) => mlp.forward(x),
            Qwen3MoeMlp::MoE(moe) => moe.forward(x),
        }
    }
}

impl Qwen3MoeDecoderLayer {
    /// Load a decoder layer.
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Qwen3MoeConfig,
        layer_idx: usize,
        dtype: DType,
        device: &Device,
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

        let mlp = if config.is_moe_layer(layer_idx) {
            Qwen3MoeMlp::MoE(Qwen3MoE::load(
                weights,
                &format!("{prefix}.mlp"),
                config,
                dtype,
            )?)
        } else {
            Qwen3MoeMlp::Dense(LlamaMLP::load(
                weights,
                &format!("{prefix}.mlp"),
                dtype,
                0,
                1,
            )?)
        };

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

    /// Create with zero weights (for testing).
    pub fn zeros(
        config: &Qwen3MoeConfig,
        layer_idx: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let llama_config = config.llama_config();
        let self_attn = LlamaAttention::zeros(&llama_config, dtype, device, layer_idx)?;

        let mlp = if config.is_moe_layer(layer_idx) {
            Qwen3MoeMlp::MoE(Qwen3MoE::zeros(config, dtype, device)?)
        } else {
            Qwen3MoeMlp::Dense(LlamaMLP::zeros(
                config.hidden_size,
                config.intermediate_size,
                dtype,
                device,
            )?)
        };

        let input_layernorm =
            RmsNorm::ones(config.hidden_size, config.rms_norm_eps, dtype, device)?;
        let post_attention_layernorm =
            RmsNorm::ones(config.hidden_size, config.rms_norm_eps, dtype, device)?;

        Ok(Self {
            self_attn,
            mlp,
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
        // Pre-attention layernorm + attention.
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

        // MLP/MoE + residual.
        let mlp_output = self.mlp.forward(&normed).map_err(ModelError::Candle)?;
        let hidden_states = (hidden_states + mlp_output).map_err(ModelError::Candle)?;

        Ok(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Qwen3MoeModel
// ---------------------------------------------------------------------------

/// Qwen3 MoE transformer backbone.
struct Qwen3MoeModel {
    embed_tokens: Embedding,
    layers: Vec<Qwen3MoeDecoderLayer>,
    norm: RmsNorm,
}

impl Qwen3MoeModel {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Qwen3MoeConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let embed_tokens = Embedding::load(weights, &format!("{prefix}.embed_tokens"), dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen3MoeDecoderLayer::load(
                weights,
                &format!("{prefix}.layers.{i}"),
                config,
                i,
                dtype,
                device,
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
// Qwen3MoeForCausalLM
// ---------------------------------------------------------------------------

/// Qwen3 MoE / Qwen2 MoE for causal language modeling.
pub struct Qwen3MoeForCausalLM {
    model: Qwen3MoeModel,
    lm_head: Linear,
}

impl Qwen3MoeForCausalLM {
    /// Load the full model from weights.
    pub fn load(
        weights: &ModelWeights,
        config: &Qwen3MoeConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let model = Qwen3MoeModel::load(weights, "model", config, dtype, device)?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight().clone(), None)
        } else {
            Linear::load(weights, "lm_head", dtype)?
        };

        Ok(Self { model, lm_head })
    }
}

impl crate::Model for Qwen3MoeForCausalLM {
    fn inject_lora(&mut self, adapter: &LoraAdapter) -> ModelResult<()> {
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            // Attention uses LlamaAttention — delegate.
            let attn_prefix = format!("model.layers.{}.self_attn", i);
            layer.self_attn.inject_lora(&attn_prefix, adapter)?;

            // MLP: only dense layers get LoRA (MoE expert LoRA is rare).
            if let Qwen3MoeMlp::Dense(ref mut mlp) = layer.mlp {
                let mlp_prefix = format!("model.layers.{}.mlp", i);
                mlp.inject_lora(&mlp_prefix, adapter)?;
            }
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
pub fn create_qwen3_moe(
    weights: &ModelWeights,
    config: &HfModelConfig,
    dtype: DType,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let moe_config = Qwen3MoeConfig::from_hf_config(config)?;
    let model = Qwen3MoeForCausalLM::load(weights, &moe_config, dtype, device)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 2,
            num_hidden_layers: 4,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: 8,
            tie_word_embeddings: false,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 32,
            shared_expert_intermediate_size: 48,
            norm_topk_prob: true,
            decoder_sparse_step: 2,
            mlp_only_layers: vec![],
        }
    }

    #[test]
    fn test_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Qwen3MoeForCausalLM"],
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "num_hidden_layers": 24,
                "intermediate_size": 8192,
                "vocab_size": 151936,
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "num_experts": 64,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 1408,
                "shared_expert_intermediate_size": 5632,
                "norm_topk_prob": true,
                "decoder_sparse_step": 2,
                "mlp_only_layers": [0]
            }"#,
        )
        .unwrap();

        let config = Qwen3MoeConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_experts, 64);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.moe_intermediate_size, 1408);
        assert_eq!(config.shared_expert_intermediate_size, 5632);
        assert_eq!(config.decoder_sparse_step, 2);
        assert_eq!(config.mlp_only_layers, vec![0]);
    }

    #[test]
    fn test_is_moe_layer() {
        let config = test_config();
        // decoder_sparse_step = 2: layers where (idx+1) % 2 == 0 are MoE.
        assert!(!config.is_moe_layer(0)); // (0+1)%2 = 1
        assert!(config.is_moe_layer(1)); // (1+1)%2 = 0
        assert!(!config.is_moe_layer(2)); // (2+1)%2 = 1
        assert!(config.is_moe_layer(3)); // (3+1)%2 = 0
    }

    #[test]
    fn test_is_moe_layer_with_mlp_only() {
        let mut config = test_config();
        config.mlp_only_layers = vec![1]; // Layer 1 forced to dense.
        assert!(!config.is_moe_layer(1)); // In mlp_only_layers.
        assert!(config.is_moe_layer(3)); // Still MoE.
    }

    #[test]
    fn test_sigmoid() {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0, 10.0], &device).unwrap();
        let y = tensor_sigmoid(&x).unwrap();
        let vals = y.to_vec1::<f32>().unwrap();
        assert!((vals[0] - 0.5).abs() < 0.01);
        assert!((vals[1] - 0.7311).abs() < 0.01);
        assert!((vals[2] - 0.2689).abs() < 0.01);
        assert!((vals[3] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_moe_forward_zeros() {
        let config = test_config();
        let device = Device::Cpu;
        let dtype = DType::F32;

        let moe = Qwen3MoE::zeros(&config, dtype, &device).unwrap();
        let x = Tensor::zeros((2, config.hidden_size), dtype, &device).unwrap();
        let output = moe.forward(&x).unwrap();
        assert_eq!(output.dims(), &[2, config.hidden_size]);
    }

    #[test]
    fn test_decoder_layer_moe_zeros() {
        let config = test_config();
        let device = Device::Cpu;
        let dtype = DType::F32;

        // MoE layer (index 1).
        let layer = Qwen3MoeDecoderLayer::zeros(&config, 1, dtype, &device).unwrap();
        let x = Tensor::zeros((3, config.hidden_size), dtype, &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();
        let output = layer.forward(&x, &positions, None).unwrap();
        assert_eq!(output.dims(), &[3, config.hidden_size]);
    }

    #[test]
    fn test_decoder_layer_dense_zeros() {
        let config = test_config();
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Dense layer (index 0).
        let layer = Qwen3MoeDecoderLayer::zeros(&config, 0, dtype, &device).unwrap();
        let x = Tensor::zeros((3, config.hidden_size), dtype, &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();
        let output = layer.forward(&x, &positions, None).unwrap();
        assert_eq!(output.dims(), &[3, config.hidden_size]);
    }

    #[test]
    fn test_registry() {
        let registry = crate::ModelRegistry::default_registry();
        assert!(registry.contains("Qwen3MoeForCausalLM"));
        assert!(registry.contains("Qwen2MoeForCausalLM"));
    }
}
