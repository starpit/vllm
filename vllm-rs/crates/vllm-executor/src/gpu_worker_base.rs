// SPDX-License-Identifier: Apache-2.0
//! Shared GPU worker utilities.
//!
//! Common logic used by all GPU-based `Worker` implementations (FerriteWorker,
//! TkWorkerAdapter, etc.): model path resolution, HF config parsing, and
//! memory estimation.
//!
//! Mirrors Python vLLM's `Worker` class which handles shared GPU plumbing
//! while delegating model-specific execution to `GPUModelRunner`.

use std::path::{Path, PathBuf};

use tracing::info;

use crate::error::{ExecutorError, ExecutorResult};

// ---------------------------------------------------------------------------
// Model path resolution
// ---------------------------------------------------------------------------

/// Resolve a model identifier to a local path.
///
/// Handles three cases:
/// 1. Local `.gguf` file path → returns as-is
/// 2. Local directory → returns as-is
/// 3. HuggingFace Hub model ID → downloads and returns cache path
///
/// Extracted from `FerriteWorker::resolve_model_path` for reuse across backends.
pub fn resolve_model_path(
    model_path: &str,
    hf_token: Option<&str>,
    gguf_file: Option<&str>,
) -> ExecutorResult<PathBuf> {
    let path = Path::new(model_path);

    // Local .gguf file.
    if path.is_file() && path.extension().is_some_and(|e| e == "gguf") {
        return Ok(path.to_path_buf());
    }

    // Local directory.
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }

    // Network-free local-cache fast path. `hf_hub::Cache` resolves the
    // refs/<revision> → snapshots/<commit>/<file> indirection on disk
    // without any HTTP. If `config.json` (sharded safetensors) or a
    // sibling `.gguf` is already present in the cache, we skip the
    // ~65ms TLS handshake + ETag check that the Api path performs even
    // for fully cached models. Network builder still runs as a
    // fallback for first-time pulls or stale caches.
    if gguf_file.is_none() {
        let cache_repo = hf_hub::Cache::from_env().model(model_path.to_string());
        if let Some(config_path) = cache_repo.get("config.json")
            && let Some(model_dir) = config_path.parent().map(|p| p.to_path_buf())
        {
            info!(
                "Using cached model: {} (HF cache, no network)",
                model_dir.display()
            );
            return Ok(model_dir);
        }
    }

    info!("Downloading model from HuggingFace Hub: {}", model_path);
    let mut builder = hf_hub::api::sync::ApiBuilder::from_env();
    if let Some(token) = hf_token {
        builder = builder.with_token(Some(token.to_string()));
    }
    let api = builder
        .build()
        .map_err(|e| ExecutorError::WorkerInit(format!("failed to build HF API: {e}")))?;
    let repo = api.model(model_path.to_string());

    // GGUF download: explicit filename or auto-detect from repo.
    let gguf_filename = gguf_file.map(String::from).or_else(|| {
        if !model_path.to_ascii_uppercase().contains("GGUF") {
            return None;
        }
        let info = repo.info().ok()?;
        let mut gguf_files: Vec<_> = info
            .siblings
            .iter()
            .filter(|s| s.rfilename.ends_with(".gguf"))
            .collect();
        if gguf_files.is_empty() {
            return None;
        }
        for pattern in &["Q4_K_M", "Q4_K_S", "Q4_K", "Q4_0", "Q8_0"] {
            if let Some(f) = gguf_files.iter().find(|s| s.rfilename.contains(pattern)) {
                return Some(f.rfilename.clone());
            }
        }
        gguf_files.sort_by(|a, b| a.rfilename.cmp(&b.rfilename));
        Some(gguf_files[0].rfilename.clone())
    });
    if let Some(ref gguf_file) = gguf_filename {
        info!("Downloading GGUF file: {gguf_file}");
        let gguf_path = repo.get(gguf_file).map_err(|e| {
            ExecutorError::WorkerInit(format!("failed to download GGUF {gguf_file}: {e}"))
        })?;
        let _ = repo.get("tokenizer.json");
        let _ = repo.get("tokenizer_config.json");
        return Ok(gguf_path);
    }

    let config_path = repo
        .get("config.json")
        .map_err(|e| ExecutorError::WorkerInit(format!("failed to download config.json: {e}")))?;
    let model_dir = config_path.parent().unwrap().to_path_buf();

    let _ = repo.get("tokenizer.json");
    let _ = repo.get("tokenizer_config.json");

    // Download weight files.
    if repo.get("model.safetensors").is_ok() {
        return Ok(model_dir);
    }
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let index = vllm_model::weight::SafeTensorsIndex::from_file(&index_path)
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to parse index: {e}")))?;
        let sorted_shards = index.shard_files();
        let total = sorted_shards.len();

        let needed: Vec<&String> = sorted_shards
            .iter()
            .filter(|s| !model_dir.join(s).exists())
            .collect();

        if needed.is_empty() {
            info!("All {total} shard files already cached");
        } else {
            info!(
                "Downloading {} of {total} shard files (up to 8 in parallel)",
                needed.len()
            );

            let multi = indicatif::MultiProgress::new();
            const MAX_PARALLEL: usize = 8;
            let repo = &repo;
            let multi = &multi;

            for chunk in needed.chunks(MAX_PARALLEL) {
                let results: Vec<ExecutorResult<()>> = std::thread::scope(|s| {
                    let handles: Vec<_> = chunk
                        .iter()
                        .map(|shard| {
                            let bar = multi.add(indicatif::ProgressBar::new(0));
                            s.spawn(move || {
                                repo.download_with_progress(shard, bar)
                                    .map(|_| ())
                                    .map_err(|e| {
                                        ExecutorError::WorkerInit(format!(
                                            "failed to download {shard}: {e}"
                                        ))
                                    })
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                for result in results {
                    result?;
                }
            }
        }
        return Ok(model_dir);
    }

    Err(ExecutorError::WorkerInit(format!(
        "no safetensors weights found for {model_path}"
    )))
}

// ---------------------------------------------------------------------------
// LlamaConfig parsing from HuggingFace config.json
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
