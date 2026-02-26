// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Stack initialization: wires together worker → executor → engine → server.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::info;
use vllm_config::{SchedulerConfig, SchedulerPolicy};
use vllm_engine::core_client::InprocClient;
use vllm_engine::engine_core::EngineCoreConfig;
use vllm_executor::candle_worker::{CandleWorker, CandleWorkerConfig};
use vllm_executor::uniproc::UniProcExecutor;
use vllm_executor::worker::Worker;
use vllm_model::weight::HfModelConfig;
use vllm_serve::chat_template::ChatTemplate;
use vllm_serve::engine::AsyncEngine;
use vllm_serve::tokenizer::Tokenizer;

use candle_core::DType;

use crate::args::ServeArgs;

/// Fully initialized stack ready to serve requests.
#[allow(dead_code)]
pub struct InitializedStack {
    /// The async engine (wraps engine core + tokenizer).
    pub engine: Arc<AsyncEngine>,
    /// Model name for display.
    pub model_name: String,
    /// Maximum model length.
    pub max_model_len: usize,
}

/// Initialize the full vLLM stack from CLI arguments.
///
/// Sequence:
/// 1. Create CandleWorkerConfig from args
/// 2. Init device, load model
/// 3. Read HfModelConfig for max_position_embeddings, num_layers, etc.
/// 4. Determine available memory, compute num_gpu_blocks
/// 5. Initialize cache
/// 6. Wrap worker in UniProcExecutor
/// 7. Build EngineCoreConfig, create InprocClient
/// 8. Load tokenizer from model_dir
/// 9. Create AsyncEngine
/// 10. Return InitializedStack
pub fn initialize_stack(args: &ServeArgs) -> Result<InitializedStack> {
    let model_path = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;

    // Extract model name before moving model_path into the worker config.
    // .into_owned() converts Cow<'_, str> to String, consuming the borrow of
    // model_path so it can be moved into the worker config below.
    let model_name = extract_model_name(&model_path).into_owned();

    // 1. Build worker config.
    let worker_config = CandleWorkerConfig {
        model_path,
        device_str: args.device.clone(),
        dtype: args.dtype.clone(),
        hf_token: args.hf_token.clone(),
        cache_dir: None,
        block_size: args.block_size,
    };

    // 2. Create worker, init device, load model.
    let mut worker = CandleWorker::new(worker_config);
    worker
        .init_device()
        .context("failed to initialize device")?;
    worker.load_model().context("failed to load model")?;

    // 3. Read model config.
    let hf_config = worker
        .hf_config()
        .context("model config not available after load")?
        .clone();
    let max_model_len = args
        .max_model_len
        .or(hf_config.max_position_embeddings)
        .unwrap_or(4096);
    let num_layers = hf_config.num_hidden_layers.unwrap_or(1);

    info!(
        "Model: {}, max_model_len={}, num_layers={}",
        model_name, max_model_len, num_layers
    );

    // 4. Determine memory and compute blocks.
    let available_memory = worker
        .determine_available_memory()
        .context("failed to determine available memory")?;

    let block_size = args.block_size;
    let model_dtype = worker.resolved_dtype().unwrap_or(DType::F32);
    let num_gpu_blocks = compute_num_blocks(available_memory, block_size, &hf_config, model_dtype);
    info!(
        "Available memory: {:.1} GB, num_gpu_blocks={}",
        available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        num_gpu_blocks
    );

    // 5. Initialize cache.
    worker
        .initialize_cache(num_gpu_blocks, 0)
        .context("failed to initialize cache")?;

    // 6. Wrap in UniProcExecutor (pre-initialized — skip init sequence).
    let model_dir = worker.model_dir().map(|p| p.to_path_buf());
    let executor = UniProcExecutor::new_pre_initialized(Box::new(worker));

    // 7. Build engine config and create InprocClient.
    //    Extract EOS token ID from model config if available.
    let eos_token_id = hf_config.extra.get("eos_token_id").and_then(|v| {
        // eos_token_id can be a single int or an array — take the first.
        if let Some(id) = v.as_u64() {
            Some(id as u32)
        } else if let Some(arr) = v.as_array() {
            arr.first().and_then(|v| v.as_u64()).map(|id| id as u32)
        } else {
            None
        }
    });
    if let Some(eos) = eos_token_id {
        info!("EOS token ID: {}", eos);
    }

    let engine_config = EngineCoreConfig {
        scheduler_config: SchedulerConfig {
            max_num_batched_tokens: max_model_len.min(8192),
            max_num_seqs: args.max_num_seqs,
            policy: SchedulerPolicy::Fcfs,
            enable_chunked_prefill: true,
            ..Default::default()
        },
        max_model_len,
        num_gpu_blocks,
        block_size,
        engine_index: 0,
        async_scheduling: false,
        use_spec_decode: false,
        eos_token_id,
    };

    let client = Box::new(InprocClient::new(engine_config, Box::new(executor)));

    // 8. Try to load tokenizer and chat template from model directory.
    let engine = if let Some(ref dir) = model_dir {
        match try_load_tokenizer(dir) {
            Ok(tok) => {
                info!("Tokenizer loaded from {}", dir.display());
                let tokenizer = Arc::new(tok);

                // Try to load chat template from tokenizer_config.json.
                let chat_template = try_load_chat_template(dir);

                if let Some(tpl) = chat_template {
                    info!("Chat template loaded from tokenizer_config.json");
                    AsyncEngine::with_tokenizer_and_template(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                        Arc::new(tpl),
                    )
                } else {
                    info!("No chat template found, using plain concatenation");
                    AsyncEngine::with_tokenizer(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                    )
                }
            }
            Err(e) => {
                info!("No tokenizer found ({}), running without", e);
                AsyncEngine::new(client, model_name.clone(), max_model_len)
            }
        }
    } else {
        AsyncEngine::new(client, model_name.clone(), max_model_len)
    };

    // 9. Return the stack.
    Ok(InitializedStack {
        engine: Arc::new(engine),
        model_name,
        max_model_len,
    })
}

/// Try to load a HuggingFace tokenizer from a model directory.
fn try_load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        anyhow::bail!("tokenizer.json not found in {}", model_dir.display());
    }
    Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))
}

/// Try to load a chat template from `tokenizer_config.json` in the model dir.
fn try_load_chat_template(model_dir: &Path) -> Option<ChatTemplate> {
    let config_path = model_dir.join("tokenizer_config.json");
    match ChatTemplate::from_tokenizer_config(&config_path) {
        Ok(Some(tpl)) => Some(tpl),
        Ok(None) => None,
        Err(e) => {
            info!("Failed to parse chat template: {e}");
            None
        }
    }
}

/// Compute the number of KV cache blocks from available memory.
fn compute_num_blocks(
    available_bytes: usize,
    block_size: usize,
    hf_config: &HfModelConfig,
    dtype: DType,
) -> usize {
    let num_layers = hf_config.num_hidden_layers.unwrap_or(1);
    let num_kv_heads = hf_config.num_kv_heads().unwrap_or(0);
    let head_dim = hf_config.head_dim().unwrap_or(0);

    // Each block holds block_size tokens of KV for all layers.
    // KV per token per layer = 2 * num_kv_heads * head_dim * sizeof(dtype)
    let elem_bytes = vllm_model::tensor::dtype_size(dtype);
    let bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * elem_bytes;
    let bytes_per_block = block_size * num_layers * bytes_per_token_per_layer;

    if bytes_per_block == 0 {
        return 1024; // Fallback.
    }

    // Use 90% of available memory for KV cache.
    let cache_memory = (available_bytes as f64 * 0.9) as usize;
    let num_blocks = cache_memory / bytes_per_block;

    // At least 16 blocks.
    num_blocks.max(16)
}

/// Extract a human-friendly model name from a path or HF ID.
fn extract_model_name(model_path: &str) -> std::borrow::Cow<'_, str> {
    // For HF IDs like "meta-llama/Llama-3.2-1B", return the full ID.
    // For local paths, return the last directory component.
    if model_path.contains('/') && !model_path.starts_with('/') {
        // Looks like an HF model ID — borrow directly, no allocation needed.
        std::borrow::Cow::Borrowed(model_path)
    } else {
        std::borrow::Cow::Owned(
            std::path::Path::new(model_path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| model_path.to_string()),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_model_name_hf_id() {
        assert_eq!(
            extract_model_name("meta-llama/Llama-3.2-1B"),
            "meta-llama/Llama-3.2-1B"
        );
    }

    #[test]
    fn test_extract_model_name_local_path() {
        assert_eq!(extract_model_name("/home/user/models/my-model"), "my-model");
    }

    #[test]
    fn test_compute_num_blocks_reasonable() {
        let config = HfModelConfig {
            num_hidden_layers: Some(32),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            hidden_size: Some(4096),
            ..Default::default()
        };
        let blocks = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, DType::F32);
        assert!(blocks >= 16);
    }

    #[test]
    fn test_compute_num_blocks_f16_more_blocks() {
        // F16 uses half the bytes per element → should yield ~2x as many blocks.
        let config = HfModelConfig {
            num_hidden_layers: Some(32),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            hidden_size: Some(4096),
            ..Default::default()
        };
        let blocks_f32 = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, DType::F32);
        let blocks_f16 = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, DType::F16);
        assert!(blocks_f16 > blocks_f32);
        // F16 should give approximately 2x the blocks.
        assert!((blocks_f16 as f64 / blocks_f32 as f64 - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_compute_num_blocks_zero_dim() {
        let config = HfModelConfig::default();
        let blocks = compute_num_blocks(1024, 16, &config, DType::F32);
        // Should fall back to 1024.
        assert_eq!(blocks, 1024);
    }
}
