// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! MLX model trait and registry.
//!
//! Provides [`MlxModel`] — the core abstraction for all MLX-based model
//! architectures, and [`MlxModelRegistry`] — maps HuggingFace architecture
//! names to MLX model constructors.

pub mod commandr;
pub mod deepseek_v2;
pub mod gemma2;
pub mod gemma3;
pub mod llama;
pub mod mixtral;
pub mod phi3;
pub mod quantized_llama;
pub mod qwen3_moe;

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;

use crate::cache::MlxKvCache;
use vllm_model::weight::HfModelConfig;

/// Core trait for all MLX model architectures.
///
/// A model takes token IDs and positions, produces logits over the vocabulary.
/// All operations are lazy — the caller must call `eval()` on the result to
/// materialize the entire forward pass as fused Metal command buffers.
pub trait MlxModel: Send {
    /// Run the model forward pass.
    ///
    /// * `input_ids` — token IDs, shape `[num_tokens]`
    /// * `positions` — position indices, shape `[num_tokens]`
    /// * `kv_cache` — per-layer KV cache. The model updates this in place.
    ///
    /// Returns logits of shape `[num_tokens, vocab_size]` (lazy — not yet evaluated).
    fn forward(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> mlx_rs::error::Result<Array>;

    /// Number of transformer layers in this model.
    fn num_layers(&self) -> usize;
}

/// Factory function type for constructing an MLX model.
pub type MlxModelFactory =
    fn(
        model_dir: &Path,
        config: &HfModelConfig,
        dtype: mlx_rs::Dtype,
    ) -> Result<Box<dyn MlxModel>, Box<dyn std::error::Error + Send + Sync>>;

/// Registry mapping HuggingFace architecture names to MLX model constructors.
pub struct MlxModelRegistry {
    models: HashMap<String, MlxModelFactory>,
    /// Separate map for quantized model factories (checked first when config has
    /// a `"quantization"` field). Mirrors the candle `gguf_models` pattern.
    quantized_models: HashMap<String, MlxModelFactory>,
}

impl MlxModelRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            quantized_models: HashMap::new(),
        }
    }

    /// Create the default registry with built-in model architectures.
    pub fn default_registry() -> Self {
        let mut registry = Self::new();
        // Non-quantized.
        registry.register("LlamaForCausalLM", llama::create_mlx_llama);
        registry.register("MistralForCausalLM", llama::create_mlx_llama);
        registry.register("Qwen2ForCausalLM", llama::create_mlx_llama);
        registry.register("Qwen3ForCausalLM", llama::create_mlx_llama);
        // Quantized (same archs, different factories).
        registry.register_quantized(
            "LlamaForCausalLM",
            quantized_llama::create_mlx_quantized_llama,
        );
        registry.register_quantized(
            "MistralForCausalLM",
            quantized_llama::create_mlx_quantized_llama,
        );
        registry.register_quantized(
            "Qwen2ForCausalLM",
            quantized_llama::create_mlx_quantized_llama,
        );
        registry.register_quantized(
            "Qwen3ForCausalLM",
            quantized_llama::create_mlx_quantized_llama,
        );
        // Gemma v1
        registry.register("GemmaForCausalLM", gemma2::create_mlx_gemma);
        registry.register_quantized("GemmaForCausalLM", gemma2::create_mlx_quantized_gemma);
        // Gemma2
        registry.register("Gemma2ForCausalLM", gemma2::create_mlx_gemma2);
        registry.register_quantized("Gemma2ForCausalLM", gemma2::create_mlx_quantized_gemma2);
        // Gemma3 (per-head QK norms, per-layer RoPE theta, no softcapping)
        registry.register("Gemma3ForCausalLM", gemma3::create_mlx_gemma3);
        registry.register_quantized("Gemma3ForCausalLM", gemma3::create_mlx_quantized_gemma3);
        // Phi-3 (fused qkv_proj + gate_up_proj)
        registry.register("Phi3ForCausalLM", phi3::create_mlx_phi3);
        registry.register_quantized("Phi3ForCausalLM", phi3::create_mlx_quantized_phi3);
        // DeepSeek V2/V3 (MLA attention + MoE)
        registry.register("DeepseekV2ForCausalLM", deepseek_v2::create_mlx_deepseek_v2);
        registry.register_quantized(
            "DeepseekV2ForCausalLM",
            deepseek_v2::create_mlx_quantized_deepseek_v2,
        );
        // Qwen3 MoE — LLaMA-like attention + MoE with sigmoid-gated shared expert
        registry.register("Qwen3MoeForCausalLM", qwen3_moe::create_mlx_qwen3_moe);
        registry.register("Qwen2MoeForCausalLM", qwen3_moe::create_mlx_qwen3_moe);
        registry.register_quantized(
            "Qwen3MoeForCausalLM",
            qwen3_moe::create_mlx_quantized_qwen3_moe,
        );
        registry.register_quantized(
            "Qwen2MoeForCausalLM",
            qwen3_moe::create_mlx_quantized_qwen3_moe,
        );
        // Mixtral — LLaMA-like attention + MoE (all layers), no shared expert
        registry.register("MixtralForCausalLM", mixtral::create_mlx_mixtral);
        registry.register_quantized("MixtralForCausalLM", mixtral::create_mlx_quantized_mixtral);
        // Command R (Cohere) — CohereLayerNorm, parallel attn+MLP, logit scaling,
        // interleaved RoPE
        registry.register("CohereForCausalLM", commandr::create_mlx_commandr);
        registry.register_quantized("CohereForCausalLM", commandr::create_mlx_quantized_commandr);
        // Kimi K2.5 — text backbone is DeepSeek V2/V3 (K2.5 factories strip
        // "language_model." weight prefix)
        registry.register("KimiK25ForCausalLM", deepseek_v2::create_mlx_kimi_k25);
        registry.register_quantized(
            "KimiK25ForCausalLM",
            deepseek_v2::create_mlx_quantized_kimi_k25,
        );
        registry
    }

    /// Register a model factory for an architecture name.
    pub fn register(&mut self, arch: &str, factory: MlxModelFactory) {
        self.models.insert(arch.to_string(), factory);
    }

    /// Register a quantized model factory for an architecture name.
    pub fn register_quantized(&mut self, arch: &str, factory: MlxModelFactory) {
        self.quantized_models.insert(arch.to_string(), factory);
    }

    /// Look up a model factory by architecture name.
    ///
    /// If `quantized` is true, checks the quantized registry first, falling
    /// back to the non-quantized registry.
    pub fn get_factory(&self, arch: &str, quantized: bool) -> Option<MlxModelFactory> {
        if quantized && let Some(f) = self.quantized_models.get(arch) {
            return Some(*f);
        }
        self.models.get(arch).copied()
    }

    /// Look up a non-quantized model factory by architecture name.
    pub fn get(&self, arch: &str) -> Option<MlxModelFactory> {
        self.models.get(arch).copied()
    }

    /// Check if an architecture is supported (quantized or not).
    pub fn contains(&self, arch: &str) -> bool {
        self.models.contains_key(arch) || self.quantized_models.contains_key(arch)
    }

    /// Check if a quantized factory exists for an architecture.
    pub fn contains_quantized(&self, arch: &str) -> bool {
        self.quantized_models.contains_key(arch)
    }

    /// Iterate over supported architecture names (non-quantized).
    pub fn architectures(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(|s| s.as_str())
    }
}

impl Default for MlxModelRegistry {
    fn default() -> Self {
        Self::default_registry()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_default() {
        let registry = MlxModelRegistry::default_registry();
        assert!(registry.contains("LlamaForCausalLM"));
        assert!(registry.contains("MistralForCausalLM"));
        assert!(registry.contains("Qwen2ForCausalLM"));
        assert!(registry.contains("GemmaForCausalLM"));
        assert!(registry.contains("Gemma2ForCausalLM"));
        assert!(registry.contains("Phi3ForCausalLM"));
        assert!(registry.contains("Qwen3MoeForCausalLM"));
        assert!(registry.contains("Qwen2MoeForCausalLM"));
        assert!(registry.contains_quantized("Gemma2ForCausalLM"));
        assert!(registry.contains_quantized("Phi3ForCausalLM"));
        assert!(registry.contains_quantized("Qwen3MoeForCausalLM"));
        assert!(registry.contains_quantized("Qwen2MoeForCausalLM"));
        assert!(registry.contains("MixtralForCausalLM"));
        assert!(registry.contains_quantized("MixtralForCausalLM"));
        assert!(!registry.contains("GPT2ForCausalLM"));
    }

    #[test]
    fn test_registry_architectures() {
        let registry = MlxModelRegistry::default_registry();
        let archs: Vec<&str> = registry.architectures().collect();
        assert!(archs.contains(&"LlamaForCausalLM"));
    }
}
