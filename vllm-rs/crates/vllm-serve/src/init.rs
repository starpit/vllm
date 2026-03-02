// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Stack initialization: wires together worker → executor → engine → server.
//!
//! This module provides the programmatic API for initializing the full vLLM
//! inference stack. No CLI dependency needed — just construct a [`VllmConfig`]
//! and call [`initialize_stack`].

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::info;
use vllm_config::{SchedulerConfig, SchedulerPolicy};
use vllm_engine::core_client::InprocClient;
use vllm_engine::engine_core::EngineCoreConfig;
use vllm_executor::candle_worker::{CandleWorker, CandleWorkerConfig};
use vllm_executor::uniproc::UniProcExecutor;
use vllm_executor::worker::Worker;
use vllm_model::weight::HfModelConfig;

#[cfg(feature = "chat-template")]
use crate::chat_template::ChatTemplate;
use crate::engine::AsyncEngine;
use crate::tokenizer::Tokenizer;

#[cfg(feature = "metal")]
use vllm_mlx::worker::{MlxWorker, MlxWorkerConfig};

use candle_core::DType;

/// Configuration for initializing the vLLM inference stack.
///
/// This is the programmatic API — no CLI dependency needed.
pub struct VllmConfig {
    /// Model path or HuggingFace model ID (required).
    pub model: String,
    /// Device: "cpu", "cuda:N", "metal", or "auto" (auto-detect best GPU).
    pub device: String,
    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    pub dtype: String,
    /// Maximum model context length (None = use config.json).
    pub max_model_len: Option<usize>,
    /// Maximum number of concurrent sequences.
    pub max_num_seqs: usize,
    /// KV cache block size in tokens.
    pub block_size: usize,
    /// Fraction of GPU memory to use for KV cache (0.0–1.0).
    pub gpu_memory_utilization: f64,
    /// HuggingFace token for gated models.
    pub hf_token: Option<String>,
    /// Specific GGUF filename to download from a HuggingFace repo.
    pub gguf_file: Option<String>,
    /// Speculative model type (e.g. "ngram"). None = disabled.
    pub speculative_model: Option<String>,
    /// Number of speculative tokens to propose per step.
    pub num_speculative_tokens: usize,
    /// Maximum n-gram size for prompt lookup.
    pub ngram_prompt_lookup_max: usize,
    /// Minimum n-gram size for prompt lookup.
    pub ngram_prompt_lookup_min: usize,
    /// LoRA adapter path (local directory or HF repo ID). None = disabled.
    pub lora_adapter: Option<String>,
    /// Pooling strategy for embeddings: "auto", "last", "cls", "mean".
    pub pooling_strategy: String,
    /// Whether to disable async scheduling (overlap GPU/CPU work).
    /// Default false — async scheduling is enabled by default.
    pub disable_async_scheduling: bool,
    /// Runner type: "generate" (default) or "pooling".
    pub runner: String,
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            device: "auto".to_string(),
            dtype: "auto".to_string(),
            max_model_len: None,
            max_num_seqs: 256,
            block_size: 16,
            gpu_memory_utilization: 0.9,
            hf_token: None,
            gguf_file: None,
            speculative_model: None,
            num_speculative_tokens: 5,
            ngram_prompt_lookup_max: 4,
            ngram_prompt_lookup_min: 1,
            lora_adapter: None,
            pooling_strategy: "auto".to_string(),
            disable_async_scheduling: false,
            runner: "generate".to_string(),
        }
    }
}

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

/// Check if the metal (MLX) feature is active and the device allows it.
#[cfg(feature = "metal")]
fn should_use_mlx(device: &str) -> bool {
    matches!(device, "auto" | "metal")
}

/// Result of worker creation: the worker plus metadata needed for init.
type WorkerCreationResult = (
    Box<dyn Worker>,
    HfModelConfig,
    Option<std::path::PathBuf>,
    DType,
);

/// Create the appropriate worker based on backend selection.
///
/// Returns `(worker, hf_config, model_dir, dtype)`.
fn create_worker(config: &VllmConfig, model_path: String) -> Result<WorkerCreationResult> {
    let is_pooling = config.runner == "pooling";

    // Try MLX backend first when metal feature is enabled.
    #[cfg(feature = "metal")]
    if should_use_mlx(&config.device) {
        info!("Using MLX backend (Apple Silicon GPU)");
        let mlx_config = MlxWorkerConfig {
            model_path: model_path.clone(),
            dtype: config.dtype.clone(),
            hf_token: config.hf_token.clone(),
            cache_dir: None,
            block_size: config.block_size,
            lora_adapter: config.lora_adapter.clone(),
            pooling_strategy: config.pooling_strategy.clone(),
            is_pooling,
        };

        let mut worker = MlxWorker::new(mlx_config);
        worker
            .init_device()
            .context("failed to initialize MLX device")?;
        worker.load_model().context("failed to load MLX model")?;

        let hf_config = worker
            .hf_config()
            .context("model config not available after MLX load")?
            .clone();
        let model_dir = worker.model_dir().map(|p| p.to_path_buf());

        // Map MLX dtype to candle DType for compute_num_blocks.
        // We resolve via the string representation to avoid depending on mlx_rs directly.
        let model_dtype = match config.dtype.as_str() {
            "f32" | "float32" => DType::F32,
            "bf16" | "bfloat16" => DType::BF16,
            _ => DType::F16, // Default: f16 for Metal
        };

        return Ok((Box::new(worker), hf_config, model_dir, model_dtype));
    }

    // Candle backend (CPU/CUDA/candle-Metal).
    info!("Using Candle backend");
    let worker_config = CandleWorkerConfig {
        model_path,
        device_str: config.device.clone(),
        dtype: config.dtype.clone(),
        hf_token: config.hf_token.clone(),
        cache_dir: None,
        block_size: config.block_size,
        gguf_file: config.gguf_file.clone(),
        lora_adapter: config.lora_adapter.clone(),
        pooling_strategy: config.pooling_strategy.clone(),
        is_pooling,
    };

    let mut worker = CandleWorker::new(worker_config);
    worker
        .init_device()
        .context("failed to initialize device")?;
    worker.load_model().context("failed to load model")?;

    let hf_config = worker
        .hf_config()
        .context("model config not available after load")?
        .clone();
    let model_dir = worker.model_dir().map(|p| p.to_path_buf());
    let model_dtype = worker.resolved_dtype().unwrap_or(DType::F32);

    Ok((Box::new(worker), hf_config, model_dir, model_dtype))
}

/// Initialize cache on the worker and compute block counts.
fn init_cache(
    mut worker: Box<dyn Worker>,
    block_size: usize,
    hf_config: &HfModelConfig,
    model_dtype: DType,
    gpu_memory_utilization: f64,
) -> Result<(Box<dyn Worker>, usize, usize)> {
    let available_memory = worker
        .determine_available_memory()
        .context("failed to determine available memory")?;
    let num_gpu_blocks = compute_num_blocks(
        available_memory,
        block_size,
        hf_config,
        model_dtype,
        gpu_memory_utilization,
    );
    worker
        .initialize_cache(num_gpu_blocks, 0)
        .context("failed to initialize cache")?;
    Ok((worker, available_memory, num_gpu_blocks))
}

/// Initialize the full vLLM stack from a [`VllmConfig`].
///
/// Sequence:
/// 1. Create worker config from args
/// 2. Init device, load model
/// 3. Read HfModelConfig for max_position_embeddings, num_layers, etc.
/// 4. Determine available memory, compute num_gpu_blocks
/// 5. Initialize cache
/// 6. Wrap worker in UniProcExecutor
/// 7. Build EngineCoreConfig, create InprocClient
/// 8. Load tokenizer from model_dir
/// 9. Create AsyncEngine
/// 10. Return InitializedStack
pub fn initialize_stack(config: &VllmConfig) -> Result<InitializedStack> {
    let init_start = Instant::now();
    let model_path = config.model.clone();

    // Extract model name before moving model_path into the worker config.
    let model_name = extract_model_name(&model_path).into_owned();

    // Decide backend: MLX (Metal) or Candle (CPU/CUDA).
    let (mut worker, hf_config, model_dir, model_dtype) = create_worker(config, model_path)?;

    // Log resolved architecture.
    if let Some(arch) = worker.architecture() {
        info!("Resolved model architecture: {}", arch);
    }

    // Take the tokenizer that was loaded in parallel during load_model().
    // This avoids a redundant parse of tokenizer.json later.
    let preloaded_tokenizer = worker.take_preloaded_tokenizer();

    let max_model_len = config
        .max_model_len
        .or(hf_config.max_position_embeddings)
        .unwrap_or(4096);
    let num_layers = hf_config.num_hidden_layers.unwrap_or(1);

    info!(
        "Model: {}, max_model_len={}, num_layers={}",
        model_name, max_model_len, num_layers
    );

    let (worker, available_memory, num_gpu_blocks) = init_cache(
        worker,
        config.block_size,
        &hf_config,
        model_dtype,
        config.gpu_memory_utilization,
    )?;

    info!(
        "Available memory: {:.1} GB, gpu_memory_utilization={}, num_gpu_blocks={}",
        available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        config.gpu_memory_utilization,
        num_gpu_blocks
    );

    let kv_cache_tokens = num_gpu_blocks * config.block_size;
    info!("KV cache size: {} tokens", kv_cache_tokens);
    info!(
        "Maximum concurrency for {} tokens per request: {:.2}x",
        max_model_len,
        kv_cache_tokens as f64 / max_model_len as f64
    );

    // 6. Wrap in UniProcExecutor (pre-initialized — skip init sequence).
    let executor = UniProcExecutor::new_pre_initialized(worker);

    // 7. Build engine config and create InprocClient.
    //    Extract EOS token ID from model config if available.
    // Parse all EOS token IDs from config.json. Models like LLaMA 3 have
    // multiple: [128001 (<|end_of_text|>), 128008 (<|eom_id|>), 128009 (<|eot_id|>)].
    // Python vLLM checks the primary via eos_token_id and adds the rest to
    // stop_token_ids; we check all of them in check_stop_criteria.
    let eos_token_ids: Vec<u32> = hf_config
        .extra
        .get("eos_token_id")
        .map(|v| {
            if let Some(id) = v.as_u64() {
                vec![id as u32]
            } else if let Some(arr) = v.as_array() {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|id| id as u32))
                    .collect()
            } else {
                vec![]
            }
        })
        .unwrap_or_default();
    if !eos_token_ids.is_empty() {
        info!("EOS token IDs: {:?}", eos_token_ids);
    }

    let engine_config = EngineCoreConfig {
        scheduler_config: SchedulerConfig {
            max_num_batched_tokens: max_model_len.min(8192),
            max_num_seqs: config.max_num_seqs,
            policy: SchedulerPolicy::Fcfs,
            enable_chunked_prefill: true,
            ..Default::default()
        },
        max_model_len,
        num_gpu_blocks,
        block_size: config.block_size,
        engine_index: 0,
        async_scheduling: true,
        use_spec_decode: config.speculative_model.is_some(),
        ngram_proposer_config: if config.speculative_model.as_deref() == Some("ngram") {
            Some(vllm_engine::ngram::NgramProposerConfig {
                num_speculative_tokens: config.num_speculative_tokens,
                max_ngram_size: config.ngram_prompt_lookup_max,
                min_ngram_size: config.ngram_prompt_lookup_min,
            })
        } else {
            None
        },
        eos_token_ids,
        is_pooling: config.runner == "pooling",
    };

    let client = Box::new(InprocClient::new(engine_config, Box::new(executor)));

    // 8. Try to load tokenizer and chat template from model directory.
    //
    // Use the tokenizer that was pre-loaded in parallel during load_model()
    // when available, avoiding a redundant parse of tokenizer.json.
    let loaded_tokenizer = preloaded_tokenizer
        .map(Tokenizer::from_hf_tokenizer)
        .or_else(|| {
            model_dir
                .as_ref()
                .and_then(|dir| match try_load_tokenizer(dir) {
                    Ok(tok) => Some(tok),
                    Err(e) => {
                        info!("No tokenizer found ({}), running without", e);
                        None
                    }
                })
        });

    let engine = if let Some(tok) = loaded_tokenizer {
        let dir_for_log = model_dir
            .as_ref()
            .map(|d| d.display().to_string())
            .unwrap_or_else(|| "<preloaded>".to_string());
        info!("Tokenizer loaded from {}", dir_for_log);
        let tokenizer = Arc::new(tok);

        // Try to load chat template from tokenizer_config.json.
        #[cfg(feature = "chat-template")]
        {
            let chat_template = model_dir
                .as_ref()
                .and_then(|dir| try_load_chat_template(dir));
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
                AsyncEngine::with_tokenizer(client, model_name.clone(), max_model_len, tokenizer)
            }
        }
        #[cfg(not(feature = "chat-template"))]
        {
            info!("No chat template found, using plain concatenation");
            AsyncEngine::with_tokenizer(client, model_name.clone(), max_model_len, tokenizer)
        }
    } else {
        AsyncEngine::new(client, model_name.clone(), max_model_len)
    };

    // 9. Enable async scheduling unless disabled. Configure pooling mode.
    let mut engine = engine;
    if !config.disable_async_scheduling {
        engine.set_async_scheduling(true);
    }
    if config.runner == "pooling" {
        engine.set_is_pooling(true);
        info!("Runner: pooling mode (embedding requests go through scheduler)");
    }

    // 10. Configure multimodal support if the model has a vision_config.
    if let Some(vision_config) = hf_config.extra.get("vision_config") {
        // Detect model type for architecture-specific defaults.
        let is_qwen2_vl = hf_config.architectures.iter().any(|a| {
            a == "Qwen2VLForConditionalGeneration" || a == "Qwen2_5_VLForConditionalGeneration"
        });

        let image_size = vision_config
            .get("image_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(if is_qwen2_vl { 392 } else { 224 }) as usize;
        let patch_size = vision_config
            .get("patch_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(14) as usize;
        let num_patches = if patch_size > 0 {
            (image_size / patch_size).pow(2)
        } else {
            256
        };

        // Qwen2-VL uses <|image_pad|> token (151655) instead of image_token_index.
        let image_token_index = if is_qwen2_vl {
            hf_config
                .extra
                .get("image_token_id")
                .and_then(|v| v.as_u64())
                .unwrap_or(151655) as u32
        } else {
            hf_config
                .extra
                .get("image_token_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(255999) as u32
        };

        // For Qwen2-VL, compute tokens per image from spatial_merge_size.
        let mm_tokens_per_image = if is_qwen2_vl {
            let spatial_merge = vision_config
                .get("spatial_merge_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize;
            let grid = image_size / patch_size;
            let merged_grid = grid / spatial_merge;
            merged_grid * merged_grid
        } else {
            hf_config
                .extra
                .get("mm_tokens_per_image")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(num_patches)
        };

        info!(
            "Multimodal config: image_token_id={}, tokens_per_image={}, image_size={}",
            image_token_index, mm_tokens_per_image, image_size,
        );
        engine.set_multimodal_config(image_token_index, mm_tokens_per_image, image_size);
        if is_qwen2_vl {
            engine.set_mm_model_type("qwen2_vl");
        }
    }

    info!(
        "init engine (load model, create kv cache) took {:.2} seconds",
        init_start.elapsed().as_secs_f64()
    );

    // 11. Return the stack.
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
#[cfg(feature = "chat-template")]
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
    gpu_memory_utilization: f64,
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

    let utilization = gpu_memory_utilization.clamp(0.0, 1.0);
    let cache_memory = (available_bytes as f64 * utilization) as usize;
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
        let blocks = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, DType::F32, 0.9);
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
        let blocks_f32 = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, DType::F32, 0.9);
        let blocks_f16 = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, DType::F16, 0.9);
        assert!(blocks_f16 > blocks_f32);
        // F16 should give approximately 2x the blocks.
        assert!((blocks_f16 as f64 / blocks_f32 as f64 - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_compute_num_blocks_zero_dim() {
        let config = HfModelConfig::default();
        let blocks = compute_num_blocks(1024, 16, &config, DType::F32, 0.9);
        // Should fall back to 1024.
        assert_eq!(blocks, 1024);
    }

    #[test]
    fn test_compute_num_blocks_utilization_half() {
        let config = HfModelConfig {
            num_hidden_layers: Some(32),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            hidden_size: Some(4096),
            ..Default::default()
        };
        let blocks_90 = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, DType::F16, 0.9);
        let blocks_50 = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, DType::F16, 0.5);
        // 0.5 should produce roughly 0.5/0.9 ≈ 55% of the blocks from 0.9.
        let ratio = blocks_50 as f64 / blocks_90 as f64;
        assert!(ratio > 0.5 && ratio < 0.65, "ratio was {ratio}");
    }

    #[test]
    fn test_vllm_config_default() {
        let config = VllmConfig::default();
        assert_eq!(config.device, "auto");
        assert_eq!(config.dtype, "auto");
        assert_eq!(config.max_num_seqs, 256);
        assert_eq!(config.block_size, 16);
        assert!((config.gpu_memory_utilization - 0.9).abs() < f64::EPSILON);
        assert!(config.model.is_empty());
    }
}
