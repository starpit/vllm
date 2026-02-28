// SPDX-License-Identifier: Apache-2.0
//! Model registry mapping architecture names to constructors.
//!
//! Port of: `vllm/model_executor/models/registry.py`

use std::collections::HashMap;

use candle_core::Device;
use vllm_model::ModelResult;
use vllm_model::gguf::GgufFile;
use vllm_model::weight::HfModelConfig;

use crate::ModelFactory;

/// Factory function type for constructing a model from a GGUF file.
pub type GgufModelFactory = fn(
    gguf: &mut GgufFile,
    config: &HfModelConfig,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>>;

/// Registry that maps HuggingFace architecture names to model factory functions.
///
/// Architecture names come from the `config.json` `"architectures"` field,
/// e.g., `"LlamaForCausalLM"`, `"MistralForCausalLM"`.
pub struct ModelRegistry {
    models: HashMap<&'static str, ModelFactory>,
    /// GGUF model factories, keyed by GGUF `general.architecture` value.
    gguf_models: HashMap<&'static str, GgufModelFactory>,
}

impl ModelRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            gguf_models: HashMap::new(),
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
        // --- SafeTensors factories ---

        // LLaMA family
        self.register("LlamaForCausalLM", crate::llama::create_llama);
        // LLaMA 2/3 use the same architecture name
        // Mistral is architecturally identical to LLaMA (same code path in vLLM Python)
        self.register("MistralForCausalLM", crate::llama::create_llama);
        // Qwen2 — architecturally identical to LLaMA but with QKV bias
        // and different rope_theta default
        self.register("Qwen2ForCausalLM", crate::qwen2::create_qwen2);
        // Qwen3 — same as Qwen2/LLaMA (no attention bias)
        self.register("Qwen3ForCausalLM", crate::llama::create_llama);
        // Phi-3 inherits directly from LLaMA in Python vLLM
        self.register("Phi3ForCausalLM", crate::llama::create_llama);
        // Gemma 2 — GELU activation, GemmaRMSNorm, 4 norms per layer,
        // attention/logit soft capping, embedding normalization
        self.register("Gemma2ForCausalLM", crate::gemma2::create_gemma2);
        // Gemma 3 — like Gemma2 but with per-head QK norms, per-layer RoPE theta,
        // sliding_window_pattern, and no softcapping
        self.register("Gemma3ForCausalLM", crate::gemma3::create_gemma3);
        // DeepSeek V2/V3 — MLA attention + MoE + YaRN RoPE
        self.register(
            "DeepseekV2ForCausalLM",
            crate::deepseek_v2::create_deepseek_v2,
        );
        // Qwen3 MoE — LLaMA-like attention + MoE with sigmoid-gated shared expert
        self.register("Qwen3MoeForCausalLM", crate::qwen3_moe::create_qwen3_moe);
        // Qwen2 MoE — same architecture as Qwen3 MoE (with QKV bias instead of QK norms)
        self.register("Qwen2MoeForCausalLM", crate::qwen3_moe::create_qwen3_moe);
        // Command R (Cohere) — CohereLayerNorm, parallel attn+MLP, logit scaling,
        // interleaved RoPE
        self.register("CohereForCausalLM", crate::commandr::create_commandr);
        // Kimi K2.5 — text backbone is DeepSeek V2/V3 (config unwrapped + weights
        // prefix-stripped by worker before calling factory)
        self.register("KimiK25ForCausalLM", crate::deepseek_v2::create_deepseek_v2);

        // --- GGUF factories (keyed by GGUF general.architecture value) ---
        self.register_gguf("llama", crate::quantized_llama::create_llama_gguf);
        // Mistral/Phi GGUF files use "llama" architecture internally.
    }

    /// Register a model factory for the given architecture name.
    pub fn register(&mut self, arch: &'static str, factory: ModelFactory) {
        self.models.insert(arch, factory);
    }

    /// Register a GGUF model factory for the given GGUF architecture.
    pub fn register_gguf(&mut self, arch: &'static str, factory: GgufModelFactory) {
        self.gguf_models.insert(arch, factory);
    }

    /// Look up a model factory by architecture name.
    pub fn get(&self, arch: &str) -> Option<&ModelFactory> {
        self.models.get(arch)
    }

    /// Look up a GGUF model factory by GGUF architecture name.
    pub fn get_gguf(&self, arch: &str) -> Option<&GgufModelFactory> {
        self.gguf_models.get(arch)
    }

    /// Check if an architecture is supported.
    pub fn contains(&self, arch: &str) -> bool {
        self.models.contains_key(arch)
    }

    /// Check if a GGUF architecture is supported.
    pub fn contains_gguf(&self, arch: &str) -> bool {
        self.gguf_models.contains_key(arch)
    }

    /// List all registered architecture names.
    pub fn architectures(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.models.keys().copied()
    }

    /// List all registered GGUF architecture names.
    pub fn gguf_architectures(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.gguf_models.keys().copied()
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

    #[test]
    fn test_gguf_registry_default() {
        let registry = ModelRegistry::default_registry();
        assert!(registry.contains_gguf("llama"));
        assert!(!registry.contains_gguf("nonexistent"));
    }

    #[test]
    fn test_gguf_registry_get() {
        let registry = ModelRegistry::default_registry();
        assert!(registry.get_gguf("llama").is_some());
        assert!(registry.get_gguf("nonexistent").is_none());
    }

    #[test]
    fn test_gguf_architectures() {
        let registry = ModelRegistry::default_registry();
        let archs: Vec<&'static str> = registry.gguf_architectures().collect();
        assert!(archs.contains(&"llama"));
    }
}
