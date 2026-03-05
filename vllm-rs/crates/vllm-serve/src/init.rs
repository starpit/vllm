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
use vllm_config::{CudaGraphConfig, SchedulerConfig, SchedulerPolicy};
use vllm_engine::core_client::InprocClient;
use vllm_engine::engine_core::EngineCoreConfig;
use vllm_executor::candle_worker::{CandleWorker, CandleWorkerConfig};
use vllm_executor::multinode::MultiNodeExecutor;
use vllm_executor::parallel::ResolvedParallelConfig;
use vllm_executor::threadpool::ThreadPoolExecutor;
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
    /// Number of GPUs for tensor parallelism (1 = single GPU, no TP).
    pub tensor_parallel_size: usize,
    /// Number of nodes for multi-node TP (1 = single node).
    pub num_nodes: usize,
    /// This node's rank (0 = master).
    pub node_rank: usize,
    /// Master address for multi-node NCCL rendezvous.
    pub master_addr: String,
    /// Master port for multi-node NCCL rendezvous.
    pub master_port: u16,
    /// Whether to disable async scheduling (overlap GPU/CPU work).
    /// Default false — async scheduling is enabled by default.
    pub disable_async_scheduling: bool,
    /// Runner type: "generate" (default) or "pooling".
    pub runner: String,
    /// CUDA graph configuration. When `Some`, CUDA graphs may be captured
    /// for decode-step acceleration.
    pub cuda_graph_config: Option<CudaGraphConfig>,
    /// Whether prefix caching is enabled (KV cache reuse for shared prompts).
    /// Default: true.
    pub enable_prefix_caching: bool,
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
            tensor_parallel_size: 1,
            num_nodes: 1,
            node_rank: 0,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            disable_async_scheduling: false,
            runner: "generate".to_string(),
            cuda_graph_config: None,
            enable_prefix_caching: true,
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

/// Result of [`initialize_stack_sync`]: the raw engine components for
/// synchronous (offline) use — no async channels, no background step loop.
/// Used by [`LLM`](crate::llm::LLM) to match Python's direct engine path.
pub struct InitializedSyncStack {
    /// In-process engine core client (owns scheduler + executor).
    pub client: InprocClient,
    /// Optional tokenizer for encoding prompts / decoding outputs.
    pub tokenizer: Option<Arc<Tokenizer>>,
    /// Optional chat template for formatting chat messages.
    pub chat_template: Option<ChatTemplate>,
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
            enable_prefix_caching: config.enable_prefix_caching,
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

    // Try wgpu backend when explicitly requested.
    #[cfg(feature = "wgpu")]
    if config.device == "wgpu" {
        info!("Using WebGPU backend");
        let mut worker = vllm_wgpu::worker_impl::WgpuBackendWorker::new(model_path);
        worker
            .init_device()
            .context("failed to initialize wgpu device")?;
        worker.load_model().context("failed to load wgpu model")?;

        let hf_config = worker
            .hf_config()
            .context("model config not available after wgpu load")?
            .clone();
        let model_dir = worker.model_dir().map(|p| p.to_path_buf());
        let model_dtype = DType::F16;

        return Ok((Box::new(worker), hf_config, model_dir, model_dtype));
    }

    // Try the purpose-built CUDA backend when feature is enabled and device is CUDA.
    #[cfg(feature = "cuda-backend")]
    if config.device.starts_with("cuda") || config.device == "auto" {
        use vllm_executor::cuda_worker::{CudaWorker, CudaWorkerConfig};

        // Parse device ID (e.g. "cuda:1" → 1, "cuda" → 0, "auto" → 0).
        let device_id = if config.device.starts_with("cuda:") {
            config.device[5..].parse::<i32>().unwrap_or(0)
        } else {
            0
        };

        info!("Using vllm-cuda backend (device={})", device_id);
        let cuda_config = CudaWorkerConfig {
            model_path: model_path.clone(),
            dtype: config.dtype.clone(),
            hf_token: config.hf_token.clone(),
            block_size: config.block_size,
            device_id,
        };

        let mut worker = CudaWorker::new(cuda_config);
        worker
            .init_device()
            .context("failed to initialize CUDA device")?;
        worker.load_model().context("failed to load CUDA model")?;

        let hf_config = worker
            .hf_config()
            .context("model config not available after CUDA load")?
            .clone();
        let model_dir = worker.model_dir().map(|p| p.to_path_buf());
        let model_dtype = worker.resolved_candle_dtype();

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
        tp_rank: 0,
        tp_world_size: 1,
        cuda_graph_config: config.cuda_graph_config.clone(),
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
///
/// On CPU, `gpu_memory_utilization` is ignored and a default of 50% is used
/// instead (matching Python vLLM's `DEFAULT_CPU_MEM_UTILIZATION`). This
/// prevents allocating most of system RAM for KV cache on CPU-only hosts.
fn init_cache(
    mut worker: Box<dyn Worker>,
    block_size: usize,
    hf_config: &HfModelConfig,
    model_dtype: DType,
    gpu_memory_utilization: f64,
    device: &str,
) -> Result<(Box<dyn Worker>, usize, usize, f64)> {
    let available_memory = worker
        .determine_available_memory()
        .context("failed to determine available memory")?;

    // CPU: ignore gpu_memory_utilization (default 0.9 is for GPU VRAM).
    // Use 50% of system RAM, matching Python vLLM's DEFAULT_CPU_MEM_UTILIZATION.
    let is_cpu = device == "cpu"
        || (device == "auto" && !cfg!(feature = "cuda") && !cfg!(feature = "metal"));
    let utilization = if is_cpu {
        const DEFAULT_CPU_MEM_UTILIZATION: f64 = 0.5;
        info!(
            "CPU device: using {:.0}% of system memory for KV cache (override with --gpu-memory-utilization)",
            DEFAULT_CPU_MEM_UTILIZATION * 100.0
        );
        DEFAULT_CPU_MEM_UTILIZATION
    } else {
        gpu_memory_utilization
    };

    let num_gpu_blocks = compute_num_blocks(
        available_memory,
        block_size,
        hf_config,
        model_dtype,
        utilization,
    );
    worker
        .initialize_cache(num_gpu_blocks, 0)
        .context("failed to initialize cache")?;
    Ok((worker, available_memory, num_gpu_blocks, utilization))
}

#[allow(dead_code)]
struct InitializedCore {
    client: InprocClient,
    tokenizer: Option<Arc<Tokenizer>>,
    model_name: String,
    max_model_len: usize,
    model_dir: Option<std::path::PathBuf>,
    hf_config: HfModelConfig,
}

/// Common initialization: worker → cache → executor → InprocClient → tokenizer.
///
/// Shared by [`initialize_stack`] (async server path) and
/// [`initialize_stack_sync`] (sync LLM path).
fn initialize_core(config: &VllmConfig) -> Result<InitializedCore> {
    let model_path = config.model.clone();
    let model_name = extract_model_name(&model_path).into_owned();

    let (mut worker, hf_config, model_dir, model_dtype) = create_worker(config, model_path)?;

    if let Some(arch) = worker.architecture() {
        info!("Resolved model architecture: {}", arch);
    }

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

    let (mut worker, available_memory, num_gpu_blocks, effective_utilization) = init_cache(
        worker,
        config.block_size,
        &hf_config,
        model_dtype,
        config.gpu_memory_utilization,
        &config.device,
    )?;

    info!(
        "Available memory: {:.1} GB, memory_utilization={}, num_gpu_blocks={}",
        available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        effective_utilization,
        num_gpu_blocks
    );

    let kv_cache_tokens = num_gpu_blocks * config.block_size;
    info!("KV cache size: {} tokens", kv_cache_tokens);
    info!(
        "Maximum concurrency for {} tokens per request: {:.2}x",
        max_model_len,
        kv_cache_tokens as f64 / max_model_len as f64
    );

    #[cfg(feature = "metrics")]
    {
        let m = crate::metrics::VllmMetrics::global();
        m.gpu_cache_blocks_total.set(num_gpu_blocks as i64);
    }

    worker
        .compile_or_warm_up_model()
        .context("failed to compile or warm up model")?;

    let executor = UniProcExecutor::new_pre_initialized(worker);

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

    let use_async_scheduling = !config.disable_async_scheduling;
    let enable_prefix_caching = config.enable_prefix_caching;

    let engine_config = EngineCoreConfig {
        scheduler_config: SchedulerConfig {
            max_num_batched_tokens: max_model_len.min(8192),
            max_num_seqs: config.max_num_seqs,
            policy: SchedulerPolicy::Fcfs,
            enable_chunked_prefill: true,
            async_scheduling: Some(use_async_scheduling),
            ..Default::default()
        },
        max_model_len,
        num_gpu_blocks,
        block_size: config.block_size,
        engine_index: 0,
        async_scheduling: use_async_scheduling,
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
        enable_prefix_caching,
    };

    let client = InprocClient::new(engine_config, Box::new(executor));

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

    let tokenizer = loaded_tokenizer.map(|tok| {
        info!("Tokenizer loaded");
        Arc::new(tok)
    });

    Ok(InitializedCore {
        client,
        tokenizer,
        model_name,
        max_model_len,
        model_dir,
        hf_config,
    })
}

/// Initialize the sync stack for offline batch inference (no async overhead).
///
/// Used by [`LLM`](crate::llm::LLM). Returns an [`InprocClient`] that the
/// caller drives directly with `add_request()` + `get_output()`.
pub fn initialize_stack_sync(config: &VllmConfig) -> Result<InitializedSyncStack> {
    let init_start = Instant::now();
    let core = initialize_core(config)?;
    info!(
        "init engine (load model, create kv cache) took {:.2} seconds",
        init_start.elapsed().as_secs_f64()
    );
    let chat_template = core.model_dir.as_deref().and_then(try_load_chat_template);
    Ok(InitializedSyncStack {
        client: core.client,
        tokenizer: core.tokenizer,
        chat_template,
        model_name: core.model_name,
        max_model_len: core.max_model_len,
    })
}

pub fn initialize_stack(config: &VllmConfig) -> Result<InitializedStack> {
    let init_start = Instant::now();

    let tp_size = config.tensor_parallel_size;
    let model_name = extract_model_name(&config.model).into_owned();

    // TP > 1: multi-GPU path with NCCL.
    if tp_size > 1 {
        return initialize_stack_tp(config, model_name, init_start);
    }

    // TP=1: single-GPU path.
    let core = initialize_core(config)?;

    let client = Box::new(core.client);

    let model_name = core.model_name;
    let max_model_len = core.max_model_len;

    let engine = if let Some(tokenizer) = core.tokenizer {
        #[cfg(feature = "chat-template")]
        {
            let chat_template = core
                .model_dir
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

    let mut engine = engine;
    if !config.disable_async_scheduling {
        engine.set_async_scheduling(true);
    }
    if config.runner == "pooling" {
        engine.set_is_pooling(true);
        info!("Runner: pooling mode (embedding requests go through scheduler)");
    }

    // Configure multimodal support if the model has a vision_config.
    let hf_config = &core.hf_config;
    if let Some(vision_config) = hf_config.extra.get("vision_config") {
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

/// Multi-GPU init flow: creates N CandleWorkers (one per GPU rank), inits NCCL,
/// and wraps them in a ThreadPoolExecutor.
///
/// Each worker loads the full model weights and keeps only its shard
/// (via ColumnParallelLinear/RowParallelLinear sharding at load time).
fn initialize_stack_tp(
    config: &VllmConfig,
    model_name: String,
    init_start: Instant,
) -> Result<InitializedStack> {
    let tp_size = config.tensor_parallel_size;
    info!(
        "Tensor parallelism: {} GPUs (thread-per-GPU model)",
        tp_size
    );

    let is_pooling = config.runner == "pooling";

    // Multi-node: each node only creates workers for its local GPUs.
    let num_nodes = config.num_nodes;
    let node_rank = config.node_rank;
    let local_tp = if num_nodes > 1 {
        tp_size / num_nodes
    } else {
        tp_size
    };

    // Create worker configs (one per local GPU).
    let worker_configs: Vec<CandleWorkerConfig> = (0..local_tp)
        .map(|local_rank| {
            let global_rank = node_rank * local_tp + local_rank;
            CandleWorkerConfig {
                model_path: config.model.clone(),
                device_str: format!("cuda:{local_rank}"),
                dtype: config.dtype.clone(),
                hf_token: config.hf_token.clone(),
                cache_dir: None,
                block_size: config.block_size,
                gguf_file: config.gguf_file.clone(),
                lora_adapter: config.lora_adapter.clone(),
                pooling_strategy: config.pooling_strategy.clone(),
                is_pooling,
                tp_rank: global_rank,
                tp_world_size: tp_size,
                cuda_graph_config: config.cuda_graph_config.clone(),
            }
        })
        .collect();

    // Clean up stale NCCL shared memory segments from previous runs.
    #[cfg(all(feature = "nccl", target_os = "linux"))]
    {
        let shm_path = std::path::Path::new("/dev/shm");
        if shm_path.exists()
            && let Ok(entries) = std::fs::read_dir(shm_path)
        {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str()
                    && name.starts_with("nccl-")
                {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }

    #[cfg(not(feature = "nccl"))]
    if tp_size > 1 {
        anyhow::bail!(
            "tensor parallelism requires the `nccl` feature; \
             rebuild with --features nccl"
        );
    }

    // Initialize all workers on dedicated per-GPU threads.
    //
    // Each thread: creates CUDA device → loads model → creates NCCL comm →
    // injects process group into model layers. This ensures NCCL comms are
    // created with the correct CUDA context (NCCL ties comms to the thread
    // that created them for collective synchronization).
    //
    // After init, workers are moved back to the main thread and wrapped
    // in the ThreadPoolExecutor.
    info!(
        "Initializing {} workers on dedicated GPU threads...",
        local_tp
    );

    // Use a barrier so rank 0 downloads/loads first (caching model files),
    // then other ranks proceed (finding cached files, no lock contention).
    let download_barrier = std::sync::Arc::new(std::sync::Barrier::new(local_tp));

    // Log NCCL-relevant env vars for debuggability.
    #[cfg(feature = "nccl")]
    {
        for var in [
            "NCCL_DEBUG",
            "NCCL_SOCKET_IFNAME",
            "NCCL_SHM_DISABLE",
            "NCCL_P2P_DISABLE",
            "NCCL_IB_DISABLE",
            "NCCL_NET_GDR_LEVEL",
        ] {
            if let Ok(val) = std::env::var(var) {
                info!("NCCL env: {var}={val}");
            }
        }
    }

    #[cfg(feature = "nccl")]
    let nccl_id = {
        use vllm_kernels::nccl::NcclProcessGroup;

        let num_nodes = config.num_nodes;
        if num_nodes <= 1 {
            NcclProcessGroup::generate_id().context("failed to generate NCCL ID")?
        } else {
            use vllm_kernels::rendezvous;
            info!(
                "Multi-node TP: node_rank={}, num_nodes={}, global_world_size={}, local_tp={}",
                node_rank, num_nodes, tp_size, local_tp
            );
            if node_rank == 0 {
                rendezvous::rendezvous_master(config.master_port, num_nodes)
                    .context("NCCL rendezvous master failed")?
            } else {
                rendezvous::rendezvous_worker(&config.master_addr, config.master_port)
                    .context("NCCL rendezvous worker failed")?
            }
        }
    };

    // Spawn one thread per GPU rank. Each thread does device init + model load
    // + NCCL comm creation + injection, then sends the worker back.
    let mut rank_counter = 0usize;
    let handles: Vec<_> = worker_configs
        .into_iter()
        .map(|cfg| {
            let local_idx = rank_counter;
            rank_counter += 1;
            // Capture NCCL init params for this rank's thread.
            #[cfg(feature = "nccl")]
            let (nccl_id, global_rank, global_ws) =
                (nccl_id, node_rank * local_tp + local_idx, tp_size);
            let barrier = download_barrier.clone();

            std::thread::spawn(move || -> Result<CandleWorker> {
                let mut worker = CandleWorker::new(cfg);
                worker.init_device().context("init_device failed")?;

                // Rank 0 loads first (downloads model files to cache).
                // Other ranks wait at the barrier, then load from cache.
                if local_idx == 0 {
                    worker.load_model().context("load_model failed")?;
                    barrier.wait();
                } else {
                    barrier.wait();
                    worker.load_model().context("load_model failed")?;
                }

                // Create NCCL comm on this thread (correct CUDA context).
                #[cfg(feature = "nccl")]
                {
                    use vllm_kernels::nccl::NcclProcessGroup;

                    let device = worker
                        .device()
                        .ok_or_else(|| anyhow::anyhow!("no device after init"))?
                        .clone();
                    let group = NcclProcessGroup::new(global_rank, global_ws, nccl_id, &device)
                        .context("NCCL comm_init_rank failed")?;
                    let group: std::sync::Arc<dyn vllm_model::process_group::ProcessGroup> =
                        std::sync::Arc::new(group);
                    worker
                        .inject_tp_group(group)
                        .context("inject_tp_group failed")?;
                }

                Ok(worker)
            })
        })
        .collect();

    let mut workers: Vec<CandleWorker> = Vec::with_capacity(local_tp);
    for (rank, handle) in handles.into_iter().enumerate() {
        let worker = handle
            .join()
            .map_err(|_| anyhow::anyhow!("worker thread {rank} panicked"))?
            .with_context(|| format!("worker {rank} init failed"))?;
        workers.push(worker);
    }

    #[cfg(feature = "nccl")]
    info!("NCCL communicators initialized and injected into model layers");

    // Extract metadata from rank 0 worker.
    let hf_config = workers[0]
        .hf_config()
        .context("model config not available after load")?
        .clone();
    let model_dir = workers[0].model_dir().map(|p| p.to_path_buf());
    let model_dtype = workers[0]
        .resolved_dtype()
        .unwrap_or(candle_core::DType::F32);
    let preloaded_tokenizer = {
        // Only take tokenizer from mutable ref to rank 0.
        // We can't mutate here since workers is Vec<CandleWorker> not Vec<&mut>.
        // The tokenizer will be loaded separately below.
        None
    };

    if let Some(arch) = workers[0].architecture() {
        info!("Resolved model architecture: {}", arch);
    }

    let max_model_len = config
        .max_model_len
        .or(hf_config.max_position_embeddings)
        .unwrap_or(4096);
    let num_layers = hf_config.num_hidden_layers.unwrap_or(1);

    info!(
        "Model: {}, max_model_len={}, num_layers={}, tp={}",
        model_name, max_model_len, num_layers, tp_size
    );

    // Wrap all workers in ThreadPoolExecutor. Each worker gets a dedicated
    // OS thread, ensuring NCCL collectives can execute concurrently.
    let all_workers: Vec<Box<dyn Worker>> = workers
        .into_iter()
        .map(|w| Box::new(w) as Box<dyn Worker>)
        .collect();

    let parallel_config = ResolvedParallelConfig::tensor_parallel(tp_size, 0);
    let mut executor = ThreadPoolExecutor::new(all_workers, parallel_config);

    // Headless worker mode: node_rank > 0 enters a blocking loop receiving
    // SchedulerOutput from the master node. Never returns (until shutdown).
    if num_nodes > 1 && node_rank > 0 {
        let control_port = config.master_port + 1;
        info!(
            "Node rank {}: entering headless worker mode (control port {})",
            node_rank, control_port
        );
        crate::headless::run_headless(executor, &config.master_addr, control_port)?;
        // run_headless only returns on shutdown — exit cleanly.
        std::process::exit(0);
    }

    // Determine available memory via executor (dispatches to worker tasks,
    // which run on the correct CUDA context).
    use vllm_engine::executor::Executor;
    let memories = executor
        .determine_available_memory()
        .context("failed to determine available memory")?;
    let min_available = *memories.iter().min().unwrap_or(&0);

    let num_gpu_blocks = compute_num_blocks(
        min_available,
        config.block_size,
        &hf_config,
        model_dtype,
        config.gpu_memory_utilization,
    );

    // Initialize cache on all workers via executor (also broadcasts to remotes
    // if multi-node).
    executor
        .initialize_cache(num_gpu_blocks, 0)
        .context("failed to initialize cache")?;

    info!(
        "Available memory (min): {:.1} GB, memory_utilization={}, num_gpu_blocks={}",
        min_available as f64 / (1024.0 * 1024.0 * 1024.0),
        config.gpu_memory_utilization,
        num_gpu_blocks
    );

    let kv_cache_tokens = num_gpu_blocks * config.block_size;
    info!("KV cache size: {} tokens", kv_cache_tokens);

    #[cfg(feature = "metrics")]
    {
        let m = crate::metrics::VllmMetrics::global();
        m.gpu_cache_blocks_total.set(num_gpu_blocks as i64);
    }

    // Build engine config.
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

    let use_async_scheduling = !config.disable_async_scheduling;
    let engine_config = EngineCoreConfig {
        scheduler_config: SchedulerConfig {
            max_num_batched_tokens: max_model_len.min(8192),
            max_num_seqs: config.max_num_seqs,
            policy: SchedulerPolicy::Fcfs,
            enable_chunked_prefill: true,
            async_scheduling: Some(use_async_scheduling),
            ..Default::default()
        },
        max_model_len,
        num_gpu_blocks,
        block_size: config.block_size,
        engine_index: 0,
        async_scheduling: use_async_scheduling,
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
        enable_prefix_caching: config.enable_prefix_caching,
    };

    // Multi-node master: wrap executor to broadcast SchedulerOutput to remotes.
    let boxed_executor: Box<dyn Executor> = if num_nodes > 1 && node_rank == 0 {
        let control_port = config.master_port + 1;
        let mn = MultiNodeExecutor::accept_remotes(executor, control_port, num_nodes - 1)
            .context("failed to accept remote nodes")?;
        Box::new(mn)
    } else {
        Box::new(executor)
    };

    let client = Box::new(InprocClient::new(engine_config, boxed_executor));

    // Load tokenizer from model directory.
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

    let mut engine = engine;
    if !config.disable_async_scheduling {
        engine.set_async_scheduling(true);
    }
    if config.runner == "pooling" {
        engine.set_is_pooling(true);
    }

    info!(
        "init engine (load model, create kv cache) took {:.2} seconds",
        init_start.elapsed().as_secs_f64()
    );

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
