// SPDX-License-Identifier: Apache-2.0
//! Model registry mapping architecture names to constructors.
//!
//! Port of: `vllm/model_executor/models/registry.py`

use std::collections::HashMap;

use crate::ModelFactory;

/// Registry that maps HuggingFace architecture names to model factory functions.
///
/// Architecture names come from the `config.json` `"architectures"` field,
/// e.g., `"LlamaForCausalLM"`, `"MistralForCausalLM"`.
pub struct ModelRegistry {
    models: HashMap<&'static str, ModelFactory>,
}

impl ModelRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Create the default registry with all built-in model architectures.
    pub fn default_registry() -> Self {
        let mut registry = Self::new();
        registry.register_builtins();
        registry
    }

    /// Register all built-in model architectures.
    fn register_builtins(&mut self) {
        // LLaMA family
        self.register("LlamaForCausalLM", crate::llama::create_llama);
        // LLaMA 2/3 use the same architecture name
        // Mistral is architecturally identical to LLaMA (same code path in vLLM Python)
        self.register("MistralForCausalLM", crate::llama::create_llama);
        // Qwen2 — architecturally identical to LLaMA but with QKV bias
        // and different rope_theta default
        self.register("Qwen2ForCausalLM", crate::qwen2::create_qwen2);
        // Phi-3 inherits directly from LLaMA in Python vLLM
        self.register("Phi3ForCausalLM", crate::llama::create_llama);
        // Gemma 2 — GELU activation, GemmaRMSNorm, 4 norms per layer,
        // attention/logit soft capping, embedding normalization
        self.register("Gemma2ForCausalLM", crate::gemma2::create_gemma2);
    }

    /// Register a model factory for the given architecture name.
    pub fn register(&mut self, arch: &'static str, factory: ModelFactory) {
        self.models.insert(arch, factory);
    }

    /// Look up a model factory by architecture name.
    pub fn get(&self, arch: &str) -> Option<&ModelFactory> {
        self.models.get(arch)
    }

    /// Check if an architecture is supported.
    pub fn contains(&self, arch: &str) -> bool {
        self.models.contains_key(arch)
    }

    /// List all registered architecture names.
    pub fn architectures(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.models.keys().copied()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::default_registry()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_default() {
        let registry = ModelRegistry::default_registry();
        assert!(registry.contains("LlamaForCausalLM"));
        assert!(registry.contains("MistralForCausalLM"));
        assert!(!registry.contains("NonexistentModel"));
    }

    #[test]
    fn test_registry_architectures() {
        let registry = ModelRegistry::default_registry();
        let archs: Vec<&'static str> = registry.architectures().collect();
        assert!(archs.contains(&"LlamaForCausalLM"));
        assert!(archs.contains(&"MistralForCausalLM"));
    }

    #[test]
    fn test_registry_get() {
        let registry = ModelRegistry::default_registry();
        assert!(registry.get("LlamaForCausalLM").is_some());
        assert!(registry.get("NonexistentModel").is_none());
    }
}
