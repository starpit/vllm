// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! MLX model trait and registry.
//!
//! Provides [`MlxModel`] — the core abstraction for all MLX-based model
//! architectures, and [`MlxModelRegistry`] — maps HuggingFace architecture
//! names to MLX model constructors.

pub mod llama;

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
}

impl MlxModelRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Create the default registry with built-in model architectures.
    pub fn default_registry() -> Self {
        let mut registry = Self::new();
        registry.register("LlamaForCausalLM", llama::create_mlx_llama);
        registry.register("MistralForCausalLM", llama::create_mlx_llama);
        registry.register("Qwen2ForCausalLM", llama::create_mlx_llama);
        registry
    }

    /// Register a model factory for an architecture name.
    pub fn register(&mut self, arch: &str, factory: MlxModelFactory) {
        self.models.insert(arch.to_string(), factory);
    }

    /// Look up a model factory by architecture name.
    pub fn get(&self, arch: &str) -> Option<MlxModelFactory> {
        self.models.get(arch).copied()
    }

    /// Check if an architecture is supported.
    pub fn contains(&self, arch: &str) -> bool {
        self.models.contains_key(arch)
    }

    /// Iterate over supported architecture names.
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
        assert!(!registry.contains("GPT2ForCausalLM"));
    }

    #[test]
    fn test_registry_architectures() {
        let registry = MlxModelRegistry::default_registry();
        let archs: Vec<&str> = registry.architectures().collect();
        assert!(archs.contains(&"LlamaForCausalLM"));
    }
}
