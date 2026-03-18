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
pub mod gemma3_mm;
pub mod granite;
pub mod llama;
pub mod mixtral;
pub mod phi3;
pub mod quantized_llama;
pub mod qwen2_vl;
pub mod qwen3_moe;
pub mod qwen3_next;
pub mod siglip;

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
#[allow(unused_imports)]
use mlx_rs::ops::indexing::IndexOp;

use crate::cache::{BatchMlxLayerKvCache, MlxBatchInfo, MlxKvCache};
use vllm_model::weight::HfModelConfig;

/// Per-GDN-layer recurrent state for MLX models: `(conv_state, ssm_state)`.
pub type MlxRecurrentState = Vec<Option<(Array, Array)>>;

/// Core trait for all MLX model architectures.
///
/// A model takes token IDs and positions, produces logits over the vocabulary.
/// All operations are lazy — the caller must call `eval()` on the result to
/// materialize the entire forward pass as fused Metal command buffers.
pub trait MlxModel: Send {
    /// Inject LoRA adapter weights into this model by merging into base weights.
    ///
    /// Default implementation: no-op (model doesn't support LoRA).
    fn inject_lora(
        &mut self,
        _adapter: &crate::lora::MlxLoraAdapter,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

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
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array>;

    /// Number of transformer layers in this model.
    fn num_layers(&self) -> usize;

    /// Run the model forward pass from pre-computed embeddings (for VLM models).
    fn forward_embeds(
        &mut self,
        _inputs_embeds: &Array,
        _positions: &Array,
        _kv_cache: &mut MlxKvCache,
        _rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        Err(mlx_rs::error::Exception::custom(
            "forward_embeds not supported by this model",
        ))
    }

    /// Provide multimodal data (images) for the next forward pass.
    fn set_mm_data(&mut self, _mm_data: Option<vllm_common::MultimodalData>) {}

    /// Reset recurrent state for hybrid models (e.g., GDN linear attention).
    ///
    /// Called before each request's forward pass. Default: no-op.
    fn reset_recurrent_state(&self) {}

    /// Number of recurrent (GDN) layers in this model.
    fn num_recurrent_layers(&self) -> usize {
        0
    }

    /// Extract recurrent state from all GDN layers (takes from internal RefCells).
    fn extract_recurrent_state(&self) -> MlxRecurrentState {
        vec![]
    }

    /// Inject previously-saved recurrent state into all GDN layers.
    fn inject_recurrent_state(&self, _state: &[Option<(Array, Array)>]) {}

    /// Run forward pass and return argmax token IDs (fused greedy decode).
    ///
    /// For greedy decode, the full `[num_tokens, vocab_size]` logits tensor is
    /// never materialized — MLX fuses the lm_head matmul + argmax into a single
    /// kernel, avoiding the huge vocab-sized intermediate.
    ///
    /// Default implementation: `forward()` → `argmax(axis=-1)`. Models can
    /// override for deeper fusion.
    fn forward_greedy(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
        rope_offset: Option<i32>,
    ) -> mlx_rs::error::Result<Array> {
        let logits = self.forward(input_ids, positions, kv_cache, rope_offset)?;
        mlx_rs::ops::indexing::argmax_axis(&logits, -1, None)
    }

    /// Run a batched forward pass over concatenated tokens from multiple requests.
    ///
    /// Default implementation falls back to per-request `forward()` calls.
    /// Models that override this can batch embedding, norms, projections, and MLP
    /// across all requests, splitting only for per-request RoPE + KV cache + SDPA.
    ///
    /// * `input_ids` — flat token IDs, shape `[total_tokens]`
    /// * `positions` — flat position indices, shape `[total_tokens]`
    /// * `batch_info` — per-request lengths, offsets, and RoPE offsets
    /// * `kv_caches` — per-request KV caches, ordered matching `batch_info`
    ///
    /// Returns logits of shape `[total_tokens, vocab_size]` (lazy).
    fn forward_batch(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        batch_info: &MlxBatchInfo,
        kv_caches: &mut [MlxKvCache],
    ) -> mlx_rs::error::Result<Array> {
        // Default: per-request loop calling forward().
        let mut logit_parts = Vec::with_capacity(batch_info.num_reqs);
        #[allow(clippy::needless_range_loop)]
        for i in 0..batch_info.num_reqs {
            let start = batch_info.offsets[i] as i32;
            let len = batch_info.q_lens[i] as i32;
            let req_ids = input_ids.index(start..start + len);
            let req_pos = positions.index(start..start + len);
            let rope_offset = Some(batch_info.rope_offsets[i]);
            let logits = self.forward(&req_ids, &req_pos, &mut kv_caches[i], rope_offset)?;
            logit_parts.push(logits);
        }
        if logit_parts.len() == 1 {
            Ok(logit_parts.into_iter().next().unwrap())
        } else {
            let refs: Vec<Array> = logit_parts;
            mlx_rs::ops::concatenate_axis(&refs, 0)
        }
    }

    /// Whether this model supports the persistent batched decode path.
    ///
    /// Models that implement `forward_batch_decode` should return `true`.
    /// Default: `false`.
    fn supports_batch_decode(&self) -> bool {
        false
    }

    /// Run a batched decode forward pass using persistent batched KV caches.
    ///
    /// All requests must be decode (q_len=1). The batched caches are updated
    /// in-place with a single `slice_update` per layer — no per-request
    /// KV cache writes or write-back needed.
    ///
    /// * `input_ids` — flat token IDs, shape `[num_reqs]` (one per request)
    /// * `batch_info` — per-request lengths, offsets, and RoPE offsets
    /// * `layer_caches` — persistent batched KV caches, one per layer
    /// * `left_padding` — per layer, per request left-padding offsets
    ///
    /// Returns logits of shape `[num_reqs, vocab_size]` (lazy).
    ///
    /// Default: returns an error (model must override to support this path).
    fn forward_batch_decode(
        &mut self,
        _input_ids: &Array,
        _batch_info: &MlxBatchInfo,
        _layer_caches: &mut [BatchMlxLayerKvCache],
        _left_padding: &[Vec<usize>],
    ) -> mlx_rs::error::Result<Array> {
        Err(mlx_rs::error::Exception::custom(
            "forward_batch_decode not supported by this model",
        ))
    }

    /// Run the model backbone and return hidden states (before lm_head).
    ///
    /// Used for embedding: a single prefill pass with no KV cache, returning
    /// the transformer output before the language model head projection.
    ///
    /// Default implementation returns an error for models that haven't
    /// overridden this method.
    fn hidden_states(
        &mut self,
        _input_ids: &Array,
        _positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        Err(mlx_rs::error::Exception::custom(
            "hidden_states not supported by this model",
        ))
    }
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
    /// a `"quantization"` field). Mirrors the `gguf_models` pattern.
    quantized_models: HashMap<String, MlxModelFactory>,
    /// GPTQ model factories, keyed by HF architecture name.
    gptq_models: HashMap<String, MlxModelFactory>,
    /// AWQ model factories, keyed by HF architecture name.
    awq_models: HashMap<String, MlxModelFactory>,
    /// BitsAndBytes NF4 model factories, keyed by HF architecture name.
    bnb_models: HashMap<String, MlxModelFactory>,
}

impl MlxModelRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            quantized_models: HashMap::new(),
            gptq_models: HashMap::new(),
            awq_models: HashMap::new(),
            bnb_models: HashMap::new(),
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
        // Gemma 3 multimodal (vision-language) — SigLIP vision + projector + Gemma3 LM
        registry.register(
            "Gemma3ForConditionalGeneration",
            gemma3_mm::create_mlx_gemma3_mm,
        );
        registry.register_quantized(
            "Gemma3ForConditionalGeneration",
            gemma3_mm::create_mlx_quantized_gemma3_mm,
        );
        // Granite (IBM) — LLaMA with embedding/residual/attention/logit multipliers
        registry.register_quantized("GraniteForCausalLM", granite::create_mlx_quantized_granite);
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
        // Qwen3-Next — hybrid GDN + full attention + MoE
        registry.register("Qwen3NextForCausalLM", qwen3_next::create_mlx_qwen3_next);
        registry.register_quantized(
            "Qwen3NextForCausalLM",
            qwen3_next::create_mlx_quantized_qwen3_next,
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
        // Qwen2-VL / Qwen2.5-VL multimodal (vision-language)
        registry.register(
            "Qwen2VLForConditionalGeneration",
            qwen2_vl::create_mlx_qwen2_vl,
        );
        registry.register_quantized(
            "Qwen2VLForConditionalGeneration",
            qwen2_vl::create_mlx_quantized_qwen2_vl,
        );
        registry.register(
            "Qwen2_5_VLForConditionalGeneration",
            qwen2_vl::create_mlx_qwen25_vl,
        );
        registry.register_quantized(
            "Qwen2_5_VLForConditionalGeneration",
            qwen2_vl::create_mlx_quantized_qwen25_vl,
        );
        registry.register_quantized(
            "KimiK25ForCausalLM",
            deepseek_v2::create_mlx_quantized_kimi_k25,
        );
        // GPTQ factories — dequantize at load time, reuse standard models.
        registry.register_gptq("LlamaForCausalLM", llama::create_mlx_gptq_llama);
        registry.register_gptq("MistralForCausalLM", llama::create_mlx_gptq_llama);
        registry.register_gptq("Qwen2ForCausalLM", llama::create_mlx_gptq_qwen2);
        registry.register_gptq("Qwen3ForCausalLM", llama::create_mlx_gptq_llama);
        registry.register_gptq("Phi3ForCausalLM", llama::create_mlx_gptq_llama);
        // AWQ factories — dequantize at load time, reuse standard models.
        registry.register_awq("LlamaForCausalLM", llama::create_mlx_awq_llama);
        registry.register_awq("MistralForCausalLM", llama::create_mlx_awq_llama);
        registry.register_awq("Qwen2ForCausalLM", llama::create_mlx_awq_qwen2);
        registry.register_awq("Qwen3ForCausalLM", llama::create_mlx_awq_llama);
        registry.register_awq("Phi3ForCausalLM", llama::create_mlx_awq_llama);
        // BitsAndBytes NF4 factories — dequantize at load time, reuse standard models.
        registry.register_bnb("LlamaForCausalLM", llama::create_mlx_bnb_llama);
        registry.register_bnb("MistralForCausalLM", llama::create_mlx_bnb_llama);
        registry.register_bnb("Qwen2ForCausalLM", llama::create_mlx_bnb_qwen2);
        registry.register_bnb("Qwen3ForCausalLM", llama::create_mlx_bnb_llama);
        registry.register_bnb("Phi3ForCausalLM", llama::create_mlx_bnb_llama);
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

    /// Register a GPTQ model factory for an architecture name.
    pub fn register_gptq(&mut self, arch: &str, factory: MlxModelFactory) {
        self.gptq_models.insert(arch.to_string(), factory);
    }

    /// Register an AWQ model factory for an architecture name.
    pub fn register_awq(&mut self, arch: &str, factory: MlxModelFactory) {
        self.awq_models.insert(arch.to_string(), factory);
    }

    /// Register a BitsAndBytes NF4 model factory for an architecture name.
    pub fn register_bnb(&mut self, arch: &str, factory: MlxModelFactory) {
        self.bnb_models.insert(arch.to_string(), factory);
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

    /// Look up a GPTQ model factory by architecture name.
    pub fn get_gptq(&self, arch: &str) -> Option<MlxModelFactory> {
        self.gptq_models.get(arch).copied()
    }

    /// Look up an AWQ model factory by architecture name.
    pub fn get_awq(&self, arch: &str) -> Option<MlxModelFactory> {
        self.awq_models.get(arch).copied()
    }

    /// Look up a BitsAndBytes NF4 model factory by architecture name.
    pub fn get_bnb(&self, arch: &str) -> Option<MlxModelFactory> {
        self.bnb_models.get(arch).copied()
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
