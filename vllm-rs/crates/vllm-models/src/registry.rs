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
use crate::awq_llama::AwqModelFactory;
use crate::bnb_llama::BnbModelFactory;
use crate::gptq_llama::GptqModelFactory;

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
    /// GPTQ model factories, keyed by HF architecture name.
    gptq_models: HashMap<&'static str, GptqModelFactory>,
    /// AWQ model factories, keyed by HF architecture name.
    awq_models: HashMap<&'static str, AwqModelFactory>,
    /// BitsAndBytes NF4 model factories, keyed by HF architecture name.
    bnb_models: HashMap<&'static str, BnbModelFactory>,
}

impl ModelRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            gguf_models: HashMap::new(),
            gptq_models: HashMap::new(),
            awq_models: HashMap::new(),
            bnb_models: HashMap::new(),
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
        // Gemma 3 multimodal (vision-language) — SigLIP vision + projector + Gemma3 LM
        self.register(
            "Gemma3ForConditionalGeneration",
            crate::gemma3_mm::create_gemma3_mm,
        );
        // Granite (IBM) — LLaMA with embedding/residual/attention/logit multipliers
        self.register("GraniteForCausalLM", crate::granite::create_granite);
        // DeepSeek V2/V3 — MLA attention + MoE + YaRN RoPE
        self.register(
            "DeepseekV2ForCausalLM",
            crate::deepseek_v2::create_deepseek_v2,
        );
        // Qwen3 MoE — LLaMA-like attention + MoE with sigmoid-gated shared expert
        self.register("Qwen3MoeForCausalLM", crate::qwen3_moe::create_qwen3_moe);
        // Qwen2 MoE — same architecture as Qwen3 MoE (with QKV bias instead of QK norms)
        self.register("Qwen2MoeForCausalLM", crate::qwen3_moe::create_qwen3_moe);
        // Mixtral — LLaMA-like attention + MoE (all layers), no shared expert
        self.register("MixtralForCausalLM", crate::mixtral::create_mixtral);
        // Command R (Cohere) — CohereLayerNorm, parallel attn+MLP, logit scaling,
        // interleaved RoPE
        self.register("CohereForCausalLM", crate::commandr::create_commandr);
        // Qwen3-Next — hybrid GDN linear attention + full attention + MoE
        self.register("Qwen3NextForCausalLM", crate::qwen3_next::create_qwen3_next);
        // Kimi K2.5 — text backbone is DeepSeek V2/V3 (config unwrapped + weights
        // prefix-stripped by worker before calling factory)
        self.register("KimiK25ForCausalLM", crate::deepseek_v2::create_deepseek_v2);
        // Qwen2-VL / Qwen2.5-VL — multimodal (vision-language) models
        self.register(
            "Qwen2VLForConditionalGeneration",
            crate::qwen2_vl::create_qwen2_vl,
        );
        self.register(
            "Qwen2_5_VLForConditionalGeneration",
            crate::qwen2_vl::create_qwen25_vl,
        );

        // --- GGUF factories (keyed by GGUF general.architecture value) ---
        self.register_gguf("llama", crate::quantized_llama::create_llama_gguf);
        self.register_gguf("granite", crate::quantized_granite::create_granite_gguf);
        // Mistral/Phi GGUF files use "llama" architecture internally.
        self.register_gguf("gemma3", crate::quantized_gemma3::create_gemma3_gguf);

        // --- GPTQ factories (keyed by HF architecture name) ---
        self.register_gptq("LlamaForCausalLM", crate::gptq_llama::create_llama_gptq);
        self.register_gptq("MistralForCausalLM", crate::gptq_llama::create_llama_gptq);
        self.register_gptq("Qwen2ForCausalLM", crate::gptq_llama::create_qwen2_gptq);
        self.register_gptq("Qwen3ForCausalLM", crate::gptq_llama::create_llama_gptq);
        self.register_gptq("Phi3ForCausalLM", crate::gptq_llama::create_llama_gptq);

        // --- AWQ factories (keyed by HF architecture name) ---
        self.register_awq("LlamaForCausalLM", crate::awq_llama::create_llama_awq);
        self.register_awq("MistralForCausalLM", crate::awq_llama::create_llama_awq);
        self.register_awq("Qwen2ForCausalLM", crate::awq_llama::create_qwen2_awq);
        self.register_awq("Qwen3ForCausalLM", crate::awq_llama::create_llama_awq);
        self.register_awq("Phi3ForCausalLM", crate::awq_llama::create_llama_awq);

        // --- BitsAndBytes NF4 factories (keyed by HF architecture name) ---
        self.register_bnb("LlamaForCausalLM", crate::bnb_llama::create_llama_bnb);
        self.register_bnb("MistralForCausalLM", crate::bnb_llama::create_llama_bnb);
        self.register_bnb("Qwen2ForCausalLM", crate::bnb_llama::create_qwen2_bnb);
        self.register_bnb("Qwen3ForCausalLM", crate::bnb_llama::create_llama_bnb);
        self.register_bnb("Phi3ForCausalLM", crate::bnb_llama::create_llama_bnb);
    }

    /// Register a model factory for the given architecture name.
    pub fn register(&mut self, arch: &'static str, factory: ModelFactory) {
        self.models.insert(arch, factory);
    }

    /// Register a GGUF model factory for the given GGUF architecture.
    pub fn register_gguf(&mut self, arch: &'static str, factory: GgufModelFactory) {
        self.gguf_models.insert(arch, factory);
    }

    /// Register a GPTQ model factory for the given HF architecture.
    pub fn register_gptq(&mut self, arch: &'static str, factory: GptqModelFactory) {
        self.gptq_models.insert(arch, factory);
    }

    /// Register an AWQ model factory for the given HF architecture.
    pub fn register_awq(&mut self, arch: &'static str, factory: AwqModelFactory) {
        self.awq_models.insert(arch, factory);
    }

    /// Register a BitsAndBytes NF4 model factory for the given HF architecture.
    pub fn register_bnb(&mut self, arch: &'static str, factory: BnbModelFactory) {
        self.bnb_models.insert(arch, factory);
    }

    /// Look up a model factory by architecture name.
    pub fn get(&self, arch: &str) -> Option<&ModelFactory> {
        self.models.get(arch)
    }

    /// Look up a GGUF model factory by GGUF architecture name.
    pub fn get_gguf(&self, arch: &str) -> Option<&GgufModelFactory> {
        self.gguf_models.get(arch)
    }

    /// Look up a GPTQ model factory by HF architecture name.
    pub fn get_gptq(&self, arch: &str) -> Option<&GptqModelFactory> {
        self.gptq_models.get(arch)
    }

    /// Look up an AWQ model factory by HF architecture name.
    pub fn get_awq(&self, arch: &str) -> Option<&AwqModelFactory> {
        self.awq_models.get(arch)
    }

    /// Look up a BitsAndBytes NF4 model factory by HF architecture name.
    pub fn get_bnb(&self, arch: &str) -> Option<&BnbModelFactory> {
        self.bnb_models.get(arch)
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

    /// Check if a GPTQ architecture is supported.
    pub fn contains_gptq(&self, arch: &str) -> bool {
        self.gptq_models.contains_key(arch)
    }

    /// List all registered GPTQ architecture names.
    pub fn gptq_architectures(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.gptq_models.keys().copied()
    }

    /// Check if an AWQ architecture is supported.
    pub fn contains_awq(&self, arch: &str) -> bool {
        self.awq_models.contains_key(arch)
    }

    /// List all registered AWQ architecture names.
    pub fn awq_architectures(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.awq_models.keys().copied()
    }

    /// Check if a BitsAndBytes NF4 architecture is supported.
    pub fn contains_bnb(&self, arch: &str) -> bool {
        self.bnb_models.contains_key(arch)
    }

    /// List all registered BitsAndBytes NF4 architecture names.
    pub fn bnb_architectures(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.bnb_models.keys().copied()
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
