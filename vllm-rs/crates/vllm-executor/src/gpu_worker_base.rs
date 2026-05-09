// SPDX-License-Identifier: Apache-2.0
//! Shared GPU worker utilities.
//!
//! Common logic used by all GPU-based `Worker` implementations (FerriteWorker,
//! TkWorkerAdapter, etc.): model path resolution, HF config parsing, and
//! memory estimation.
//!
//! Mirrors Python vLLM's `Worker` class which handles shared GPU plumbing
//! while delegating model-specific execution to `GPUModelRunner`.

#[cfg(feature = "cuda")]
use crate::error::{ExecutorError, ExecutorResult};
#[cfg(feature = "cuda")]
use tracing::info;

// ---------------------------------------------------------------------------
// Model path resolution lives on `ferrite_worker::resolve_model_path` —
// this module previously held a copy that diverged. Both copies have
// been consolidated onto the ferrite_worker version (the one with
// parallel shard download + GGUF auto-detect + HF-Cache short-circuit).
// `vllm-serve` and the worker `load_model` bodies all call into the
// canonical version directly.
// ---------------------------------------------------------------------------

/// Parse a `LlamaConfig` from a HuggingFace `config.json`.
///
/// Extracted from `FerriteWorker::llama_config_from_hf` for reuse across backends.
#[cfg(feature = "cuda")]
pub fn llama_config_from_hf(
    hf: &vllm_model::weight::HfModelConfig,
) -> ExecutorResult<vllm_cuda::model::llama::LlamaConfig> {
    let hidden_size = hf
        .hidden_size
        .ok_or_else(|| ExecutorError::WorkerInit("missing hidden_size in config.json".into()))?;
    let num_attention_heads = hf
        .num_attention_heads
        .ok_or_else(|| ExecutorError::WorkerInit("missing num_attention_heads".into()))?;
    let num_kv_heads = hf.num_key_value_heads.unwrap_or(num_attention_heads);
    let head_dim = hf.head_dim.unwrap_or(hidden_size / num_attention_heads);

    let llama3_rope_scaling = hf.extra.get("rope_scaling").and_then(|rs| {
        let rope_type = rs
            .get("rope_type")
            .or_else(|| rs.get("type"))
            .and_then(|v| v.as_str())?;
        if rope_type != "llama3" {
            return None;
        }
        Some(vllm_cuda::model::llama::Llama3RopeScaling {
            factor: rs.get("factor")?.as_f64()?,
            low_freq_factor: rs
                .get("low_freq_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0),
            high_freq_factor: rs
                .get("high_freq_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(4.0),
            original_max_position_embeddings: rs
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(8192) as usize,
        })
    });

    if let Some(ref s) = llama3_rope_scaling {
        info!(
            "llama3 rope_scaling: factor={}, low_freq={}, high_freq={}, orig_max={}",
            s.factor, s.low_freq_factor, s.high_freq_factor, s.original_max_position_embeddings
        );
    }

    Ok(vllm_cuda::model::llama::LlamaConfig {
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        num_hidden_layers: hf.num_hidden_layers.unwrap_or(32),
        intermediate_size: hf.intermediate_size.unwrap_or(hidden_size * 4),
        vocab_size: hf.vocab_size.unwrap_or(32000),
        max_position_embeddings: hf.max_position_embeddings.unwrap_or(4096),
        rms_norm_eps: hf.rms_norm_eps.unwrap_or(1e-5) as f32,
        rope_theta: hf.rope_theta.unwrap_or(10000.0),
        head_dim,
        tie_word_embeddings: hf.tie_word_embeddings.unwrap_or(false),
        llama3_rope_scaling,
    })
}

// ---------------------------------------------------------------------------
// Memory estimation helpers
// ---------------------------------------------------------------------------

/// Compute available bytes for KV cache given GPU memory profile data.
///
/// Matches Python vLLM's computation:
///   available = total * utilization - weights_and_overhead - peak_activations - redundancy
pub fn compute_available_kv_bytes(
    total_memory: usize,
    weights_and_overhead: usize,
    peak_activations: usize,
    gpu_memory_utilization: f64,
) -> usize {
    let redundancy_buffer: usize = 150 * 1024 * 1024; // 150 MiB
    let non_kv_cache = weights_and_overhead + peak_activations + redundancy_buffer;
    let requested = (total_memory as f64 * gpu_memory_utilization) as usize;
    requested.saturating_sub(non_kv_cache)
}
