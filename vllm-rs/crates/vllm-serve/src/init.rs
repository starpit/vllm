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
#[cfg(any(feature = "cuda", feature = "metal"))]
use vllm_config::CudaGraphMode;
use vllm_config::{CudaGraphConfig, SchedulerConfig, SchedulerPolicy};
use vllm_engine::core_client::InprocClient;
use vllm_engine::engine_core::EngineCoreConfig;
use vllm_engine::spec_decode::{DraftModelProposerConfig, NgramProposerConfig, ProposerConfig};
use vllm_executor::uniproc::UniProcExecutor;
use vllm_executor::worker::Worker;
use vllm_model::weight::HfModelConfig;

use crate::chat_template::ChatTemplate;
use crate::engine::AsyncEngine;
use crate::tokenizer::Tokenizer;

// vllm-mlx has been removed. The metal path lives entirely in
// ferrite-metal (`FerriteWorker`); see `project_vllm_mlx_nuke_plan`.
// If you hit `ExecutorError::ArchNotSupported` at runtime, the fix is
// to add the missing arch / quantization overlay to
// `crates/ferrite-model-<arch>/configs/quantizations.json` (or land
// the matching backend Impl in `crates/ferrite-forward-macro/src/metal/`)
// — NOT to bring back MLX.

/// Configuration for initializing the vLLM inference stack.
///
/// This is the programmatic API — no CLI dependency needed.
#[derive(Debug, Clone, serde::Serialize)]
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
    /// HuggingFace token for gated models (excluded from serialization).
    #[serde(skip_serializing)]
    pub hf_token: Option<String>,
    /// Specific GGUF filename to download from a HuggingFace repo.
    pub gguf_file: Option<String>,
    /// Speculative model: `"ngram"` selects the n-gram proposer; any other
    /// value is treated as a draft-model local path or HuggingFace repo ID
    /// (wired in phase 4 of `DRAFT_SPEC_DECODE_PLAN.md`). `None` disables
    /// speculative decoding.
    pub speculative_model: Option<String>,
    /// Number of speculative tokens to propose per step.
    pub num_speculative_tokens: usize,
    /// Maximum n-gram size for prompt lookup.
    pub ngram_prompt_lookup_max: usize,
    /// Minimum n-gram size for prompt lookup.
    pub ngram_prompt_lookup_min: usize,
    /// Optional dtype override for the draft model's weights ("auto",
    /// "float16", "bfloat16", etc). `None` inherits the target's dtype.
    /// Only meaningful when `speculative_model` points at a draft model.
    pub draft_model_dtype: Option<String>,
    /// LoRA adapter path (local directory or HF repo ID). None = disabled.
    pub lora_adapter: Option<String>,
    /// Pooling strategy for embeddings: "auto", "last", "cls", "mean".
    pub pooling_strategy: String,
    /// Number of GPUs for tensor parallelism (1 = single GPU, no TP).
    pub tensor_parallel_size: usize,
    /// Number of GPU stages for pipeline parallelism (1 = no PP).
    pub pipeline_parallel_size: usize,
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
    /// Disable CUDA graph capture and run all steps eagerly.
    /// Default: false.
    pub enforce_eager: bool,
    /// CUDA graph mode: controls piecewise vs monolithic graph capture.
    /// Default: "full" (maintains current behavior).
    pub cuda_graph_mode: String,
    /// Maximum number of tokens processed in a single scheduler iteration.
    /// None = auto (min(max_model_len, 8192)).
    pub max_num_batched_tokens: Option<usize>,
    /// Benchmark cublasLt algorithms during warmup.
    pub cublas_autotune: bool,
    /// KV cache data type: "auto" (use model dtype) or "fp8_e4m3".
    pub kv_cache_dtype: String,
    /// Compute KV scales dynamically from the first forward pass.
    pub calculate_kv_scales: bool,
    /// Distributed executor backend: "auto" or "external_launcher".
    /// "external_launcher" reads RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
    /// from env and uses TCP-based NCCL init for inter-process TP.
    pub distributed_executor_backend: String,
    /// Optional chat template override.
    ///
    /// If `Some`, takes precedence over every auto-detected source
    /// (GGUF metadata `tokenizer.chat_template`, `tokenizer_config.json`,
    /// `chat_template.jinja`). Accepts either an inline Jinja template
    /// string or a filesystem path that gets read as the template body.
    /// Required for GGUFs (e.g. mmnga's Moonlight) whose converters
    /// dropped the template and whose source HF repo isn't co-located.
    pub chat_template: Option<String>,
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
            // K=2 is the empirical sweet spot for draft-model spec
            // decode on Apple Silicon when the target dominates
            // (e.g. 8B+1B). See `vllm-cli/src/args.rs` for the bench
            // distribution that led to this default.
            num_speculative_tokens: 2,
            ngram_prompt_lookup_max: 4,
            ngram_prompt_lookup_min: 1,
            draft_model_dtype: None,
            lora_adapter: None,
            pooling_strategy: "auto".to_string(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            num_nodes: 1,
            node_rank: 0,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            cuda_graph_mode: "auto".to_string(),
            disable_async_scheduling: false,
            runner: "generate".to_string(),
            cuda_graph_config: None,
            enable_prefix_caching: true,
            enforce_eager: true, // TODO: debug multi-turn — disable CUDA graphs to isolate
            max_num_batched_tokens: None,
            cublas_autotune: false,
            kv_cache_dtype: "auto".to_string(),
            calculate_kv_scales: false,
            distributed_executor_backend: "auto".to_string(),
            chat_template: None,
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
    /// KV cache block size in tokens.
    pub block_size: usize,
}

/// True iff the device string selects the Apple-silicon metal path
/// (`auto` or `metal`). Name retained as `should_use_mlx` for historical
/// continuity with the pre-`vllm-mlx`-nuke fallback chain; today there
/// is no MLX backend — the metal path lives entirely in
/// `ferrite-metal` (`FerriteWorker`).
#[cfg(feature = "metal")]
fn should_use_mlx(device: &str) -> bool {
    matches!(device, "auto" | "metal")
}

/// Resolve the full set of stop-on-generate token IDs the engine
/// should honor, merging three HuggingFace conventions:
///
/// 1. `config.json::eos_token_id` — single int or array. Always read.
/// 2. `generation_config.json::eos_token_id` — same shape, but
///    authoritative when present (HF transformers reads this first
///    at `model.generate()` time; vLLM Python mirrors that).
/// 3. `tokenizer_config.json::eos_token` — a string (or
///    `{content: ...}` object) naming the actual end-of-turn token
///    for the chat template (e.g. Llama-3-Instruct stores
///    `"<|eot_id|>"` here, distinct from `config.json::eos_token_id`
///    = 128001 which is `<|end_of_text|>`). Resolved against the
///    loaded tokenizer's vocab; missing tokenizer → skipped.
///
/// Without #3 the chat loop never stops on `<|eot_id|>` (128009) for
/// Llama-3-Instruct: `config.json::eos_token_id` is 128001, the base
/// model's end-of-document token, which the Instruct fine-tune
/// emits only after a final `<|eot_id|>`. Replicates the merge
/// `transformers.generation.utils.GenerationMixin._prepare_generation_config`
/// performs on the Python side.
fn resolve_eos_token_ids(
    hf_config: &HfModelConfig,
    model_dir: Option<&std::path::Path>,
    preloaded_tokenizer: Option<&tokenizers::Tokenizer>,
) -> Vec<u32> {
    fn extract_ids(v: &serde_json::Value) -> Vec<u32> {
        if let Some(id) = v.as_u64() {
            return vec![id as u32];
        }
        if let Some(arr) = v.as_array() {
            return arr
                .iter()
                .filter_map(|e| e.as_u64().map(|id| id as u32))
                .collect();
        }
        Vec::new()
    }

    let mut out: Vec<u32> = Vec::new();
    let push = |id: u32, out: &mut Vec<u32>| {
        if !out.contains(&id) {
            out.push(id);
        }
    };

    // (1) config.json::eos_token_id
    if let Some(v) = hf_config.extra.get("eos_token_id") {
        for id in extract_ids(v) {
            push(id, &mut out);
        }
    }

    let Some(dir) = model_dir else {
        return out;
    };

    // (2) generation_config.json::eos_token_id
    let gen_cfg_path = dir.join("generation_config.json");
    if let Ok(bytes) = std::fs::read(&gen_cfg_path)
        && let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes)
        && let Some(v) = json.get("eos_token_id")
    {
        for id in extract_ids(v) {
            push(id, &mut out);
        }
    }

    // (3) tokenizer_config.json::eos_token (string) → resolved via tokenizer.
    let tok_cfg_path = dir.join("tokenizer_config.json");
    if let (Ok(bytes), Some(tokenizer)) = (std::fs::read(&tok_cfg_path), preloaded_tokenizer)
        && let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes)
        && let Some(eos) = json.get("eos_token")
    {
        let token_str: Option<&str> = if let Some(s) = eos.as_str() {
            Some(s)
        } else {
            eos.get("content").and_then(|v| v.as_str())
        };
        if let Some(s) = token_str
            && let Some(id) = tokenizer.token_to_id(s)
        {
            push(id, &mut out);
        }
    }

    out
}

/// Result of worker creation: the worker plus metadata needed for init.
///
/// The `usize` is the KV cache element size in bytes (e.g. 2 for F16/BF16, 4 for F32).
type WorkerCreationResult = (
    Box<dyn Worker>,
    HfModelConfig,
    Option<std::path::PathBuf>,
    usize,
);

/// Create the appropriate worker based on backend selection.
///
/// Returns `(worker, hf_config, model_dir, dtype)`.
#[allow(unused_variables)]
fn create_worker(
    config: &VllmConfig,
    model_path: String,
    progress: Option<&Arc<crate::progress::StartupProgress>>,
) -> Result<WorkerCreationResult> {
    let is_pooling = config.runner == "pooling";

    // ferrite-metal is the only Apple-silicon backend. The historical
    // MLX fallback (`vllm-mlx`) was deleted because it silently masked
    // ferrite-metal gaps — a checkpoint whose arch × quantization
    // wasn't registered with ferrite-metal would silently route to MLX
    // and produce garbage from MLX-side bugs the user attributed to
    // ferrite-metal. Now the fallback is a hard error pointing at
    // exactly what to add.
    #[cfg(feature = "metal")]
    if should_use_mlx(&config.device) {
        use vllm_executor::error::ExecutorError;
        use vllm_executor::ferrite_worker::{FerriteWorker, FerriteWorkerConfig};

        let ferrite_config = FerriteWorkerConfig {
            model_path: model_path.clone(),
            dtype: config.dtype.clone(),
            hf_token: config.hf_token.clone(),
            block_size: config.block_size,
            device_id: 0,
            enforce_eager: config.enforce_eager,
            cuda_graph_mode: config
                .cuda_graph_mode
                .parse()
                .unwrap_or(CudaGraphMode::Auto),
            max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
            cuda_graph_sizes: config
                .cuda_graph_config
                .as_ref()
                .map(|c| c.capture_sizes.clone())
                .unwrap_or_default(),
            cublas_autotune: config.cublas_autotune,
            gpu_memory_utilization: config.gpu_memory_utilization,
            pooling_strategy: config.pooling_strategy.clone(),
            is_pooling,
            tp_rank: 0,
            tp_world_size: 1,
            pp_rank: 0,
            pp_size: 1,
            gguf_file: config.gguf_file.clone(),
            lora_adapter: config.lora_adapter.clone(),
            kv_cache_dtype: config.kv_cache_dtype.clone(),
            calculate_kv_scales: config.calculate_kv_scales,
            eos_token_ids: vec![],
            max_model_len: config.max_model_len,
            draft_model_path: spec_decode_draft_model_path(config),
            draft_model_dtype: config.draft_model_dtype.clone(),
        };

        let mut ferrite = FerriteWorker::new(ferrite_config);
        ferrite
            .init_device()
            .context("failed to initialize Metal device for FerriteWorker")?;
        match ferrite.load_model() {
            Ok(()) => {
                info!("Using ferrite-metal backend (Apple Silicon GPU)");
                let hf_config = ferrite
                    .hf_config()
                    .context("model config not available after ferrite-metal load")?
                    .clone();
                let model_dir = ferrite.model_dir().map(|p| p.to_path_buf());
                let dtype_elem_bytes = ferrite.resolved_dtype_elem_bytes();
                return Ok((Box::new(ferrite), hf_config, model_dir, dtype_elem_bytes));
            }
            Err(ExecutorError::ArchNotSupported(arch)) => {
                return Err(anyhow::anyhow!(
                    "ferrite-metal has no compiled variant for arch `{arch}`. \
                     The vllm-mlx fallback was removed — fix this by adding the \
                     missing (arch, quantization) combination to \
                     `crates/ferrite-model-<arch>/configs/quantizations.json`, \
                     or by adding a per-arch backend Impl under \
                     `crates/ferrite-forward-macro/src/metal/` and relaxing the \
                     matching `BackendCompat<Metal>` assert. Then rebuild with \
                     `FERRITE_MODELS=<variant-stem>`."
                ));
            }
            Err(e) => return Err(e).context("failed to load ferrite-metal model"),
        }
    }

    // Try the purpose-built CUDA backend when feature is enabled and device is CUDA.
    #[cfg(feature = "cuda")]
    if config.device.starts_with("cuda") || config.device == "auto" {
        use vllm_executor::ferrite_worker::{FerriteWorker, FerriteWorkerConfig};

        // Parse device ID (e.g. "cuda:1" → 1, "cuda" → 0, "auto" → 0).
        let device_id = if config.device.starts_with("cuda:") {
            config.device[5..].parse::<i32>().unwrap_or(0)
        } else {
            0
        };

        info!("Using vllm-cuda backend (device={})", device_id);
        let cuda_config = FerriteWorkerConfig {
            model_path: model_path.clone(),
            dtype: config.dtype.clone(),
            hf_token: config.hf_token.clone(),
            block_size: config.block_size,
            device_id,
            enforce_eager: config.enforce_eager,
            cuda_graph_mode: config
                .cuda_graph_mode
                .parse()
                .unwrap_or(CudaGraphMode::Auto),
            // Default 1024 (not 8192 like Python). Our FerriteWorker splits mixed
            // batches into a decode CUDA-graph pass + a prefill eager pass.
            // Smaller prefill chunks keep the eager pass fast (~25ms for 1024
            // tokens) while decode runs through the captured graph (~5ms).
            // Benchmarked: 1024 → 21.8 req/s vs 8192 → 12.1 req/s on Qwen2.5-3B.
            // See PREFILL_DECODE_SPLIT.md for the full analysis.
            max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
            cuda_graph_sizes: config
                .cuda_graph_config
                .as_ref()
                .map(|c| c.capture_sizes.clone())
                .unwrap_or_default(),
            cublas_autotune: config.cublas_autotune,
            gpu_memory_utilization: config.gpu_memory_utilization,
            pooling_strategy: config.pooling_strategy.clone(),
            is_pooling,
            tp_rank: 0,
            tp_world_size: 1,
            pp_rank: 0,
            pp_size: 1,
            gguf_file: config.gguf_file.clone(),
            lora_adapter: config.lora_adapter.clone(),
            kv_cache_dtype: config.kv_cache_dtype.clone(),
            calculate_kv_scales: config.calculate_kv_scales,
            eos_token_ids: vec![],
            max_model_len: config.max_model_len,
            draft_model_path: spec_decode_draft_model_path(config),
            draft_model_dtype: config.draft_model_dtype.clone(),
        };

        let mut worker = FerriteWorker::new(cuda_config);
        worker
            .init_device()
            .context("failed to initialize CUDA device")?;

        // Store progress reference in worker for layer-by-layer updates
        if let Some(pb) = progress {
            let pb_clone = pb.clone();
            worker.set_progress_callback(std::sync::Arc::new(move |msg: &str| {
                // For now, just log the message. Layer-by-layer progress
                // would require deeper integration into model loading.
                if msg.contains("layer") {
                    // Could parse layer number and update sub-progress here
                }
            }));
        }

        worker.load_model().context("failed to load CUDA model")?;

        let hf_config = worker
            .hf_config()
            .context("model config not available after CUDA load")?
            .clone();
        let model_dir = worker.model_dir().map(|p| p.to_path_buf());
        let dtype_elem_bytes = worker.resolved_dtype_elem_bytes();

        return Ok((Box::new(worker), hf_config, model_dir, dtype_elem_bytes));
    }

    // No backend available.
    anyhow::bail!(
        "No backend available for device '{}'. \
         Build with --features metal (Apple Silicon) or --features cuda (NVIDIA GPU).",
        config.device
    )
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
    dtype_elem_bytes: usize,
    gpu_memory_utilization: f64,
    device: &str,
    kv_cache_dtype: &str,
) -> Result<(Box<dyn Worker>, usize, usize, f64)> {
    let available_memory = worker
        .determine_available_memory()
        .context("failed to determine available memory")?;

    // For CUDA workers, determine_available_memory already returns the KV
    // cache budget with gpu_memory_utilization baked in (matching Python vLLM:
    // total * util - non_kv_cache). For CPU/Metal, the returned value is raw
    // free memory and we apply utilization here.
    let is_cuda = cfg!(feature = "cuda")
        && device != "cpu"
        && (device == "auto" || device.starts_with("cuda"));
    let is_cpu = device == "cpu"
        || (device == "auto" && !cfg!(feature = "cuda") && !cfg!(feature = "metal"));

    let (kv_cache_bytes, utilization) = if is_cuda {
        // FerriteWorker already computed: total * util - non_kv_cache
        (available_memory, gpu_memory_utilization)
    } else if is_cpu {
        const DEFAULT_CPU_MEM_UTILIZATION: f64 = 0.5;
        info!(
            "CPU device: using {:.0}% of system memory for KV cache (override with --gpu-memory-utilization)",
            DEFAULT_CPU_MEM_UTILIZATION * 100.0
        );
        let bytes = (available_memory as f64 * DEFAULT_CPU_MEM_UTILIZATION) as usize;
        (bytes, DEFAULT_CPU_MEM_UTILIZATION)
    } else {
        // Metal or other GPU — apply utilization to raw free memory.
        let bytes = (available_memory as f64 * gpu_memory_utilization) as usize;
        (bytes, gpu_memory_utilization)
    };

    let num_gpu_blocks = compute_num_blocks(
        kv_cache_bytes,
        block_size,
        hf_config,
        dtype_elem_bytes,
        utilization,
        kv_cache_dtype,
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

/// `--speculative-model <repo>` as a draft-model path, or `None` for
/// the n-gram proposer / no spec-decode case. Threaded into each
/// `FerriteWorkerConfig` construction so the worker can load the draft
/// alongside the target.
fn spec_decode_draft_model_path(config: &VllmConfig) -> Option<String> {
    match config.speculative_model.as_deref() {
        Some("ngram") | None => None,
        Some(path) => Some(path.to_string()),
    }
}

/// Build the speculative-decoding proposer config from a `VllmConfig`.
///
/// Three outcomes:
///   * `speculative_model == None` → `Ok(None)` (spec decode disabled).
///   * `speculative_model == Some("ngram")` → `Ok(Some(Ngram(...)))`.
///   * `speculative_model == Some(other)` → `Ok(Some(DraftModel(...)))` —
///     but [`validate_speculative_decoding`] (called earlier from each
///     `initialize_stack*` entry point) refuses to start an engine with
///     this variant until phase 4 of `DRAFT_SPEC_DECODE_PLAN.md` lands.
///
/// Today this fn returns `Err` for the draft-model branch as a defense in
/// depth in case the entry-point validator is bypassed. Phase 4 replaces
/// that `Err` with the real draft-model wiring.
fn build_proposer_config(
    config: &VllmConfig,
    max_model_len: usize,
) -> Result<Option<ProposerConfig>> {
    let Some(spec) = config.speculative_model.as_deref() else {
        return Ok(None);
    };
    if spec == "ngram" {
        return Ok(Some(ProposerConfig::Ngram(NgramProposerConfig {
            num_speculative_tokens: config.num_speculative_tokens,
            max_ngram_size: config.ngram_prompt_lookup_max,
            min_ngram_size: config.ngram_prompt_lookup_min,
            max_model_len,
        })));
    }
    // Non-"ngram" value: treat as a draft-model path / HF repo. The
    // engine logs the registration and proceeds as baseline until phase 4
    // wires the real proposer (`EngineCore::new` handles this variant by
    // leaving the proposer as `None`).
    Ok(Some(ProposerConfig::DraftModel(DraftModelProposerConfig {
        model: spec.to_string(),
        num_speculative_tokens: config.num_speculative_tokens,
        dtype: config.draft_model_dtype.clone(),
        max_model_len,
    })))
}

/// Validate `--speculative-model` against the phase guards before any
/// model load. Called at the top of each `initialize_stack*` entry.
///
/// For draft-model speculative decoding (`--speculative-model <path>`):
///   1. Vocab/tokenizer alignment (`validate_target_draft_pair`).
///   2. Metal memory budget rough-check (metal feature only).
///
/// Phase 4 of DRAFT_SPEC_DECODE_PLAN.md: the worker runs a K-step draft
/// chain after each target step and stashes the drafts on
/// `ModelRunnerOutput.draft_token_ids`; the engine forwards them to the
/// scheduler via `set_spec_token_ids`. The metal-side M>1 verify path
/// is a follow-up — until it lands, drafts are proposed but discarded
/// by the worker's next prepare_inputs, and the engine reports 0%
/// acceptance.
fn validate_speculative_decoding(config: &VllmConfig) -> Result<()> {
    let Some(spec) = config.speculative_model.as_deref() else {
        return Ok(());
    };
    if spec == "ngram" {
        return Ok(());
    }
    validate_target_draft_pair(config, spec)?;
    info!(
        "spec-decode: draft model {:?} accepted; worker-side proposer runs a K-step \
         decode chain per step.",
        spec
    );
    Ok(())
}

/// Mirror Python vLLM (vllm/config/vllm.py:709-721): force-disable async
/// scheduling for non-EAGLE spec-decode methods. Both ngram and draft-model
/// paths seed K-step drafts from step N's output for step N+1's verify;
/// async's 1-step lookahead skews that to N+2 and acceptance collapses to
/// ~0. EAGLE/MTP methods (not yet ported) re-use hidden states directly
/// and are compatible with async.
fn spec_decode_requires_sync(config: &VllmConfig) -> bool {
    config.speculative_model.is_some()
}

/// Phase-2 guards: vocab/tokenizer match + memory budget for the
/// target + draft pair.
///
/// Resolves both models to their HF-cache snapshot dirs and asserts:
///   * `target.vocab_size == draft.vocab_size` (mirrors Python vLLM's
///     SpeculativeConfig vocab-size check).
///   * `sha256(target/tokenizer.json) == sha256(draft/tokenizer.json)`
///     (the cheapest "same tokenizer" assertion that survives metadata
///     differences like padding-token whitespace).
///   * On metal builds: weight bytes + a coarse KV estimate fit under
///     `recommendedMaxWorkingSetSize`.
///
/// Files that don't exist locally trigger `resolve_model_path`'s normal
/// HF download path — so a misconfigured draft URL surfaces here, not
/// halfway through model load.
fn validate_target_draft_pair(config: &VllmConfig, draft_spec: &str) -> Result<()> {
    let target_dir = vllm_executor::ferrite_worker::resolve_model_path(
        &config.model,
        config.hf_token.as_deref(),
        config.gguf_file.as_deref(),
    )
    .with_context(|| format!("resolving target model path {:?}", config.model))?;
    let draft_dir = vllm_executor::ferrite_worker::resolve_model_path(
        draft_spec,
        config.hf_token.as_deref(),
        None, // draft never uses target's --gguf-file
    )
    .with_context(|| format!("resolving draft model path {draft_spec:?}"))?;

    let target_cfg = HfModelConfig::from_dir(&target_dir)
        .with_context(|| format!("reading target config.json at {:?}", target_dir))?;
    let draft_cfg = HfModelConfig::from_dir(&draft_dir)
        .with_context(|| format!("reading draft config.json at {:?}", draft_dir))?;

    if let (Some(tv), Some(dv)) = (target_cfg.vocab_size, draft_cfg.vocab_size)
        && tv != dv
    {
        anyhow::bail!(
            "target/draft vocab_size mismatch: target={} ({}), draft={} ({}). \
                 Spec-decode requires matching vocabularies — the draft model must \
                 emit token IDs the target can interpret. Pick a draft from the same \
                 model family (e.g. Llama-3.2-1B paired with Llama-3.2-3B).",
            tv,
            config.model,
            dv,
            draft_spec
        );
    }

    let target_tok = target_dir.join("tokenizer.json");
    let draft_tok = draft_dir.join("tokenizer.json");
    if target_tok.is_file() && draft_tok.is_file() {
        let tb = std::fs::read(&target_tok).with_context(|| format!("reading {:?}", target_tok))?;
        let db = std::fs::read(&draft_tok).with_context(|| format!("reading {:?}", draft_tok))?;
        if tb != db {
            // Byte-equal failed. The two HF families that pair well
            // for spec-decode (Llama-3.1 target + Llama-3.2 draft,
            // Qwen-2.5 target + Qwen-2.5-0.5B draft, etc.) ship
            // tokenizer.json files that are SEMANTICALLY identical
            // but serialize merges differently (older `"a b"` strings
            // vs newer `["a", "b"]` lists) or carry different
            // whitespace/version metadata. Reject only if the
            // SEMANTIC token table actually differs.
            tokenizer_semantic_mismatch_check(&tb, &db, &target_tok, &draft_tok)?;
        }
    }

    #[cfg(feature = "metal")]
    metal_memory_budget_check(config, &target_cfg, &target_dir, &draft_cfg, &draft_dir)?;
    #[cfg(not(feature = "metal"))]
    {
        let _ = (&target_cfg, &draft_cfg, &target_dir, &draft_dir);
    }
    Ok(())
}

/// Compare two `tokenizer.json` files semantically (not byte-equal).
///
/// Pairs from the same family (Llama-3.1-8B target + Llama-3.2-1B
/// draft, Qwen-2.5-7B target + Qwen-2.5-0.5B draft, etc.) ship
/// tokenizer.json files that produce identical token IDs but
/// serialize the merges differently:
///   * Older HF tokenizers write `model.merges` as
///     `["a b", "c d", ...]` (space-separated strings).
///   * Newer tokenizers write `[["a","b"], ["c","d"], ...]` (lists).
///
/// Normalize before comparing so cosmetic format drift doesn't block
/// spec-decode. The actual token IDs are identical iff
/// (`model.vocab`, `added_tokens`, `normalized(model.merges)`) match.
fn tokenizer_semantic_mismatch_check(
    target_bytes: &[u8],
    draft_bytes: &[u8],
    target_path: &Path,
    draft_path: &Path,
) -> Result<()> {
    let parse = |bytes: &[u8], path: &Path| -> Result<serde_json::Value> {
        serde_json::from_slice::<serde_json::Value>(bytes)
            .with_context(|| format!("parsing {path:?} as JSON for spec-decode tokenizer check"))
    };
    let t = parse(target_bytes, target_path)?;
    let d = parse(draft_bytes, draft_path)?;

    let normalize_merges = |v: &serde_json::Value| -> Vec<(String, String)> {
        let arr = match v.as_array() {
            Some(a) => a,
            None => return Vec::new(),
        };
        let mut out = Vec::with_capacity(arr.len());
        for m in arr {
            if let Some(s) = m.as_str() {
                if let Some((a, b)) = s.split_once(' ') {
                    out.push((a.to_string(), b.to_string()));
                }
            } else if let Some(pair) = m.as_array()
                && pair.len() == 2
                && let (Some(a), Some(b)) = (pair[0].as_str(), pair[1].as_str())
            {
                out.push((a.to_string(), b.to_string()));
            }
        }
        out
    };

    let tv = t.pointer("/model/vocab");
    let dv = d.pointer("/model/vocab");
    if tv != dv {
        anyhow::bail!(
            "target/draft tokenizer.json mismatch: model.vocab differs ({} vs {} bytes on disk). \
             Spec-decode requires identical token IDs in both models. \
             Pick a draft from the same model family as the target.",
            target_bytes.len(),
            draft_bytes.len(),
        );
    }
    let ta = t.pointer("/added_tokens");
    let da = d.pointer("/added_tokens");
    if ta != da {
        anyhow::bail!(
            "target/draft tokenizer.json mismatch: added_tokens differs. \
             Spec-decode requires identical token IDs in both models. \
             Pick a draft from the same model family as the target.",
        );
    }
    let tm = normalize_merges(
        t.pointer("/model/merges")
            .unwrap_or(&serde_json::Value::Null),
    );
    let dm = normalize_merges(
        d.pointer("/model/merges")
            .unwrap_or(&serde_json::Value::Null),
    );
    if tm != dm {
        anyhow::bail!(
            "target/draft tokenizer.json mismatch: model.merges differs (normalized {} vs {} entries). \
             Spec-decode requires identical token IDs in both models. \
             Pick a draft from the same model family as the target.",
            tm.len(),
            dm.len(),
        );
    }
    // Vocab + added_tokens + merges all match semantically — the two
    // tokenizers produce identical token IDs even though the file
    // bytes differ. Safe for spec decode.
    info!(
        "spec-decode tokenizer.json byte-differs but is semantically identical \
         (vocab + added_tokens + normalized merges match) — accepted."
    );
    Ok(())
}

/// Coarse memory-budget check for the target+draft pair on Metal.
///
/// Checks **resident weight bytes only** — KV cache is paged and sized
/// dynamically at runtime against whatever the worker reports as
/// available, so it's the wrong shape to compare against a static
/// budget here. The goal is to catch obvious misconfigurations (e.g.
/// 7B + 3B target+draft on a 24 GiB machine) before model load.
///
/// Asserts that `target_weights + draft_weights` stays under
/// `recommendedMaxWorkingSetSize * gpu_memory_utilization * 0.6`,
/// reserving ~40% for KV cache, activations, and other working-set
/// overhead.
#[cfg(feature = "metal")]
fn metal_memory_budget_check(
    config: &VllmConfig,
    target_cfg: &HfModelConfig,
    target_dir: &std::path::Path,
    draft_cfg: &HfModelConfig,
    draft_dir: &std::path::Path,
) -> Result<()> {
    let _ = (target_cfg, draft_cfg);
    let Some(total) = vllm_executor::metal_info::recommended_max_working_set_size() else {
        return Ok(()); // No Metal device — skip.
    };
    // Reserve at least 40% of the gpu-memory-utilization budget for KV +
    // activations. Empirically the runtime sizes KV against
    // `(working_set - currentAllocatedSize) * gpu_memory_utilization`,
    // so weights consuming >60% leaves KV starved.
    let weight_budget = (total as f64 * config.gpu_memory_utilization * 0.6).round() as u64;

    let target_weights = weight_bytes_on_disk(target_dir);
    let draft_weights = weight_bytes_on_disk(draft_dir);
    let total_w = target_weights + draft_weights;

    if total_w > weight_budget {
        anyhow::bail!(
            "estimated combined weight footprint {} > {} (60% of \
             recommendedMaxWorkingSetSize × gpu_memory_utilization={:.2}). \
             Breakdown: target_weights={}, draft_weights={}. \
             Either pick a smaller draft, reduce --gpu-memory-utilization, \
             or run on a machine with more unified memory.",
            human_bytes(total_w),
            human_bytes(weight_budget),
            config.gpu_memory_utilization,
            human_bytes(target_weights),
            human_bytes(draft_weights),
        );
    }
    info!(
        "spec-decode weight budget OK: target_w={} + draft_w={} = {} / {} weight budget \
         (Metal working set = {})",
        human_bytes(target_weights),
        human_bytes(draft_weights),
        human_bytes(total_w),
        human_bytes(weight_budget),
        human_bytes(total),
    );
    Ok(())
}

#[cfg(feature = "metal")]
fn weight_bytes_on_disk(dir: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|e| {
            let n = e.file_name();
            let s = n.to_string_lossy();
            s.ends_with(".safetensors")
                || s.ends_with(".gguf")
                || (s.starts_with("model") && s.ends_with(".bin"))
        })
        // `entry.metadata()` reports the *symlink's* own size; HF cache
        // entries are symlinks into the blob store, so we have to call
        // `fs::metadata(path)` to follow the link to the actual file.
        .filter_map(|e| std::fs::metadata(e.path()).ok())
        .map(|m| m.len())
        .sum()
}

#[cfg(feature = "metal")]
fn human_bytes(b: u64) -> String {
    const GIB: f64 = 1_073_741_824.0;
    const MIB: f64 = 1_048_576.0;
    let f = b as f64;
    if f >= GIB {
        format!("{:.2} GiB", f / GIB)
    } else {
        format!("{:.0} MiB", f / MIB)
    }
}

/// Common initialization: worker → cache → executor → InprocClient → tokenizer.
///
/// Handles both single-GPU (TP=1) and multi-GPU (TP>1) transparently.
/// Shared by [`initialize_stack`] (async server path) and
/// [`initialize_stack_sync`] (sync LLM path).
fn initialize_core(
    config: &VllmConfig,
    progress: Option<&Arc<crate::progress::StartupProgress>>,
) -> Result<InitializedCore> {
    let tp_size = config.tensor_parallel_size;

    // TP > 1: multi-GPU path with NCCL.
    if tp_size > 1 {
        return initialize_core_tp(config);
    }

    let model_path = config.model.clone();
    let model_name = extract_model_name(&model_path).into_owned();

    if let Some(pb) = progress {
        pb.set_stage("Loading model");
    }

    // Resolve the model dir up-front so we can load the tokenizer in
    // parallel with the worker's heavy weight upload. `try_load_tokenizer`
    // is the canonical loader (init.rs::try_load_tokenizer) and lives
    // here in vllm-serve — workers shouldn't be re-implementing it.
    // For cached models `resolve_model_path` is sub-ms; for HF download
    // it does the download once, then `worker.load_model` re-resolves
    // the same cached path (cheap). Returns `None` for `.gguf` sources
    // (no sibling tokenizer.json) — that path falls through to the
    // worker's `take_preloaded_tokenizer` (cuda/mlx GGUF builds the
    // tokenizer from file metadata via `ferrite_gguf::gguf_tokenizer`,
    // the only reason any worker still touches tokenizers).
    let prefetched_model_dir = vllm_executor::ferrite_worker::resolve_model_path(
        &model_path,
        config.hf_token.as_deref(),
        config.gguf_file.as_deref(),
    )
    .ok();
    let parallel_tokenizer_handle = prefetched_model_dir.as_ref().and_then(|dir| {
        let tok_path = dir.join("tokenizer.json");
        if !tok_path.exists() {
            return None;
        }
        Some(std::thread::spawn(move || {
            tokenizers::Tokenizer::from_file(&tok_path).ok()
        }))
    });

    let (mut worker, hf_config, model_dir, model_dtype) =
        create_worker(config, model_path, progress)?;

    if let Some(arch) = worker.architecture() {
        info!("Resolved model architecture: {}", arch);
    }

    let preloaded_tokenizer = worker
        .take_preloaded_tokenizer()
        .or_else(|| parallel_tokenizer_handle.and_then(|h| h.join().ok().flatten()));

    let max_model_len = config
        .max_model_len
        .or(hf_config.max_position_embeddings)
        .unwrap_or(4096);
    let num_layers = hf_config.num_hidden_layers.unwrap_or(1);

    info!(
        "Model: {}, max_model_len={}, num_layers={}",
        model_name, max_model_len, num_layers
    );

    if let Some(pb) = progress {
        pb.set_stage("Initializing KV cache");
    }
    let block_size = config.block_size;

    let (mut worker, available_memory, num_gpu_blocks, effective_utilization) = init_cache(
        worker,
        block_size,
        &hf_config,
        model_dtype,
        config.gpu_memory_utilization,
        &config.device,
        &config.kv_cache_dtype,
    )?;

    info!(
        "Available memory: {:.1} GB, memory_utilization={}, num_gpu_blocks={}",
        available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        effective_utilization,
        num_gpu_blocks
    );

    let kv_cache_tokens = num_gpu_blocks * block_size;
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

    if let Some(pb) = progress {
        pb.set_stage("Warming up model");
    }
    worker
        .compile_or_warm_up_model()
        .context("failed to compile or warm up model")?;

    let executor = UniProcExecutor::new_pre_initialized(worker);

    let eos_token_ids: Vec<u32> = resolve_eos_token_ids(
        &hf_config,
        model_dir.as_deref(),
        preloaded_tokenizer.as_ref(),
    );
    if !eos_token_ids.is_empty() {
        info!("EOS token IDs: {:?}", eos_token_ids);
    }

    let use_async_scheduling =
        !config.disable_async_scheduling && !spec_decode_requires_sync(config);
    let enable_prefix_caching = config.enable_prefix_caching;

    let engine_config = EngineCoreConfig {
        scheduler_config: SchedulerConfig {
            // Default 2048. Mixed batches use the unified eager path.
            // See PREFILL_DECODE_SPLIT.md for history.
            max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(2048),
            max_num_seqs: config.max_num_seqs,
            policy: SchedulerPolicy::Fcfs,
            enable_chunked_prefill: true,
            async_scheduling: Some(use_async_scheduling),
            num_lookahead_tokens: if config.speculative_model.is_some() {
                config.num_speculative_tokens
            } else {
                0
            },
            ..Default::default()
        },
        max_model_len,
        num_gpu_blocks,
        block_size,
        engine_index: 0,
        async_scheduling: use_async_scheduling,
        use_spec_decode: config.speculative_model.is_some(),
        proposer_config: build_proposer_config(config, max_model_len)?,
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

    let tokenizer = loaded_tokenizer.map(|mut tok| {
        if let Some(dir) = model_dir.as_ref() {
            patch_additional_special_tokens(&mut tok, dir);
        }
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

/// TP variant of `initialize_core`: spawns one worker per GPU, sets up NCCL,
/// profiles memory, warms up, and returns an `InitializedCore` with a
/// `ThreadPoolExecutor`-backed `InprocClient`.
fn initialize_core_tp(config: &VllmConfig) -> Result<InitializedCore> {
    #[cfg(not(feature = "nccl"))]
    {
        let _ = config;
        anyhow::bail!(
            "Tensor parallelism requires the `nccl` feature; \
             rebuild with --features nccl"
        );
    }

    #[cfg(feature = "nccl")]
    {
        use vllm_executor::ferrite_worker::{FerriteWorker, FerriteWorkerConfig};
        use vllm_executor::parallel::ResolvedParallelConfig;
        use vllm_executor::threadpool::ThreadPoolExecutor;

        let tp_size = config.tensor_parallel_size;
        let model_name = extract_model_name(&config.model).into_owned();
        let is_pooling = config.runner == "pooling";

        info!("Tensor parallelism: {} GPUs", tp_size);

        let nccl_id = vllm_cuda::NcclId::new().context("failed to generate NCCL unique ID")?;

        let worker_configs: Vec<FerriteWorkerConfig> = (0..tp_size)
            .map(|rank| FerriteWorkerConfig {
                model_path: config.model.clone(),
                dtype: config.dtype.clone(),
                hf_token: config.hf_token.clone(),
                block_size: config.block_size,
                device_id: rank as i32,
                enforce_eager: config.enforce_eager,
                max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(2048),
                cuda_graph_sizes: config
                    .cuda_graph_config
                    .as_ref()
                    .map(|c| c.capture_sizes.clone())
                    .unwrap_or_default(),
                cublas_autotune: config.cublas_autotune,
                gpu_memory_utilization: config.gpu_memory_utilization,
                pooling_strategy: config.pooling_strategy.clone(),
                is_pooling,
                tp_rank: rank,
                tp_world_size: tp_size,
                pp_rank: 0,
                pp_size: 1,
                gguf_file: config.gguf_file.clone(),
                lora_adapter: config.lora_adapter.clone(),
                kv_cache_dtype: config.kv_cache_dtype.clone(),
                calculate_kv_scales: config.calculate_kv_scales,
                cuda_graph_mode: config
                    .cuda_graph_mode
                    .parse()
                    .unwrap_or(CudaGraphMode::Auto),
                eos_token_ids: vec![],
                max_model_len: config.max_model_len,
                draft_model_path: spec_decode_draft_model_path(config),
                draft_model_dtype: config.draft_model_dtype.clone(),
            })
            .collect();

        let download_barrier = std::sync::Arc::new(std::sync::Barrier::new(tp_size));

        let handles: Vec<_> = worker_configs
            .into_iter()
            .enumerate()
            .map(|(local_rank, cfg)| {
                let barrier = download_barrier.clone();
                std::thread::spawn(move || -> Result<FerriteWorker> {
                    let mut worker = FerriteWorker::new(cfg);
                    worker.init_device().context("init_device failed")?;
                    if local_rank == 0 {
                        worker.load_model().context("load_model failed")?;
                        barrier.wait();
                    } else {
                        barrier.wait();
                        worker.load_model().context("load_model failed")?;
                    }
                    Ok(worker)
                })
            })
            .collect();

        let mut ferrite_workers: Vec<FerriteWorker> = Vec::with_capacity(tp_size);
        let mut hf_config = None;
        let mut model_dir = None;
        let mut dtype_elem_bytes: usize = 2;
        // GGUF sources carry the tokenizer in the file's metadata, not as
        // a sibling `tokenizer.json` on disk. `FerriteWorker::load_model`
        // reconstructs it and stashes it in `preloaded_tokenizer`. The
        // tp=1 path (`initialize_core`) harvests this via
        // `take_preloaded_tokenizer()` and falls back to
        // `try_load_tokenizer(dir)` only when there's no preloaded one.
        // The tp>1 path previously skipped this and went straight to
        // `try_load_tokenizer` — which returns `None` for GGUF (no
        // sibling tokenizer.json) and silently dropped the real
        // tokenizer, leaving the engine with a byte-level fallback that
        // tokenized `<|begin_of_text|>` as literal ASCII bytes (prompt
        // became 235 garbage tokens, all downstream inference followed).
        let mut preloaded_tokenizer: Option<tokenizers::Tokenizer> = None;

        for (rank, handle) in handles.into_iter().enumerate() {
            let mut worker = handle
                .join()
                .map_err(|_| anyhow::anyhow!("worker thread {rank} panicked"))?
                .with_context(|| format!("worker {rank} init failed"))?;
            if rank == 0 {
                hf_config = worker.hf_config().cloned();
                model_dir = worker.model_dir().map(|p| p.to_path_buf());
                dtype_elem_bytes = worker.resolved_dtype_elem_bytes();
                preloaded_tokenizer = worker.take_preloaded_tokenizer();
            }
            ferrite_workers.push(worker);
        }

        let hf_config = hf_config.context("model config not available after load")?;
        let max_model_len = config
            .max_model_len
            .or(hf_config.max_position_embeddings)
            .unwrap_or(4096);

        info!(
            "Model: {}, max_model_len={}, tp={}",
            model_name, max_model_len, tp_size
        );

        // NCCL init + memory profiling on persistent threads.
        let init_results: Vec<Result<(usize, FerriteWorker)>> = std::thread::scope(|s| {
            let handles: Vec<_> = ferrite_workers
                .into_iter()
                .enumerate()
                .map(|(rank, mut worker)| {
                    s.spawn(move || -> Result<(usize, FerriteWorker)> {
                        let device = worker.device_ref().expect("device not initialized");
                        unsafe {
                            vllm_cuda::driver::ctx_set_current(device.ctx).unwrap();
                        }
                        let nccl_group = vllm_cuda::NcclGroup::new(
                            rank,
                            tp_size,
                            nccl_id,
                            device.compute_stream,
                        )
                        .context("NCCL comm init failed")?;
                        worker.set_tp_group(std::sync::Arc::new(nccl_group));
                        let avail = worker.determine_available_memory().map_err(|e| {
                            anyhow::anyhow!("determine_available_memory rank {rank}: {e}")
                        })?;
                        Ok((avail, worker))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut workers: Vec<Box<dyn Worker>> = Vec::with_capacity(tp_size);
        let mut min_avail = usize::MAX;
        for res in init_results {
            let (avail, worker) = res?;
            min_avail = min_avail.min(avail);
            workers.push(Box::new(worker));
        }

        let num_gpu_blocks = compute_num_blocks(
            min_avail,
            config.block_size,
            &hf_config,
            dtype_elem_bytes,
            config.gpu_memory_utilization,
            &config.kv_cache_dtype,
        );

        for w in &mut workers {
            w.initialize_cache(num_gpu_blocks, 0)
                .context("initialize_cache")?;
        }

        info!(
            "TP: min available memory across ranks: {:.1} GB, num_gpu_blocks={}",
            min_avail as f64 / (1024.0 * 1024.0 * 1024.0),
            num_gpu_blocks,
        );

        // Warm up concurrently (NCCL collectives need all ranks).
        let warmup_results: Vec<Result<()>> = std::thread::scope(|s| {
            let handles: Vec<_> = workers
                .iter_mut()
                .enumerate()
                .map(|(rank, w)| {
                    s.spawn(move || {
                        w.compile_or_warm_up_model()
                            .with_context(|| format!("compile_or_warm_up_model rank {rank}"))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for res in warmup_results {
            res?;
        }

        let parallel_config = ResolvedParallelConfig::tensor_parallel(tp_size, 0);
        let executor = ThreadPoolExecutor::new(workers, parallel_config);

        let use_async_scheduling =
            (!config.disable_async_scheduling && !spec_decode_requires_sync(config));
        let enable_prefix_caching = config.enable_prefix_caching;

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

        let engine_config = EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(2048),
                max_num_seqs: config.max_num_seqs,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                async_scheduling: Some(use_async_scheduling),
                num_lookahead_tokens: if config.speculative_model.is_some() {
                    config.num_speculative_tokens
                } else {
                    0
                },
                ..Default::default()
            },
            max_model_len,
            num_gpu_blocks,
            block_size: config.block_size,
            engine_index: 0,
            async_scheduling: use_async_scheduling,
            use_spec_decode: config.speculative_model.is_some(),
            proposer_config: None,
            eos_token_ids,
            is_pooling: config.runner == "pooling",
            enable_prefix_caching,
        };

        let client = InprocClient::new(engine_config, Box::new(executor));

        // Prefer the GGUF-preloaded tokenizer harvested from rank 0's
        // worker above. Falls back to `try_load_tokenizer(dir)` for
        // safetensors where a sibling `tokenizer.json` lives next to
        // the weights. Mirrors `initialize_core`'s tp=1 flow.
        let tokenizer = preloaded_tokenizer
            .map(Tokenizer::from_hf_tokenizer)
            .or_else(|| {
                model_dir
                    .as_ref()
                    .and_then(|dir| try_load_tokenizer(dir).ok())
            })
            .map(|tok| {
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
}

/// Initialize the sync stack for offline batch inference (no async overhead).
///
/// Used by [`LLM`](crate::llm::LLM). Returns an [`InprocClient`] that the
/// caller drives directly with `add_request()` + `get_output()`.
pub fn initialize_stack_sync(config: &VllmConfig) -> Result<InitializedSyncStack> {
    validate_speculative_decoding(config)?;
    let init_start = Instant::now();

    // Create progress bar if logging is below INFO level
    let progress = {
        let show_progress = !tracing::enabled!(tracing::Level::INFO);
        Arc::new(crate::progress::StartupProgress::new(show_progress))
    };

    let core = initialize_core(config, Some(&progress))?;
    progress.finish();
    info!(
        "init engine (load model, create kv cache) took {:.2} seconds",
        init_start.elapsed().as_secs_f64()
    );
    let chat_template = core
        .model_dir
        .as_deref()
        .and_then(|dir| resolve_chat_template(config.chat_template.as_deref(), dir));
    Ok(InitializedSyncStack {
        client: core.client,
        tokenizer: core.tokenizer,
        chat_template,
        model_name: core.model_name,
        max_model_len: core.max_model_len,
        block_size: config.block_size,
    })
}

pub fn initialize_stack(
    config: &VllmConfig,
    progress: Option<Arc<crate::progress::StartupProgress>>,
) -> Result<InitializedStack> {
    validate_speculative_decoding(config)?;
    let init_start = Instant::now();

    // Create progress bar if not provided and logging is below INFO level
    let progress = progress.unwrap_or_else(|| {
        // Check if tracing is enabled at INFO level or higher
        let show_progress = !tracing::enabled!(tracing::Level::INFO);
        Arc::new(crate::progress::StartupProgress::new(show_progress))
    });

    let tp_size = config.tensor_parallel_size;
    let model_name = extract_model_name(&config.model).into_owned();

    // External launcher: each process is a separate rank with its own GPU.
    if config.distributed_executor_backend == "external_launcher" {
        return initialize_stack_external(config, model_name, init_start);
    }

    // Multi-node TP ("mp" backend): leader runs engine+scheduler, followers
    // run headless. TCP for control plane, NCCL for data plane.
    // Followers (node_rank > 0) should use `initialize_and_run_follower` instead.
    if (config.num_nodes > 1 || config.distributed_executor_backend == "mp")
        && config.node_rank == 0
    {
        return initialize_stack_multinode(config, model_name, init_start);
    }

    let pp_size = config.pipeline_parallel_size;

    // TP+PP or PP-only: multi-GPU path with NCCL P2P for PP and all-reduce for TP.
    if pp_size > 1 {
        return initialize_stack_tp_pp(config, model_name, init_start);
    }

    // TP > 1: multi-GPU path with NCCL (in-process, thread-per-GPU).
    if tp_size > 1 {
        return initialize_stack_tp(config, model_name, init_start);
    }

    // TP=1: single-GPU path.
    progress.set_stage("Initializing backend");
    let core = initialize_core(config, Some(&progress))?;

    progress.set_stage("Creating engine");

    let client = Box::new(core.client);

    let model_name = core.model_name;
    let max_model_len = core.max_model_len;

    let engine = if let Some(tokenizer) = core.tokenizer {
        {
            let chat_template = core
                .model_dir
                .as_ref()
                .and_then(|dir| resolve_chat_template(config.chat_template.as_deref(), dir));
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
    } else {
        AsyncEngine::new(client, model_name.clone(), max_model_len)
    };

    let mut engine = engine;
    if !config.disable_async_scheduling && !spec_decode_requires_sync(config) {
        engine.set_async_scheduling(true);
    }
    if config.runner == "pooling" {
        engine.set_is_pooling(true);
        info!("Runner: pooling mode (embedding requests go through scheduler)");
    }

    // Configure multimodal support — the per-arch declaration lives in
    // ferrite-vision::MmMetadata (collected via ferrite-forward's
    // inventory). vllm-serve names no arch; if no MM arch claims the
    // current HF architectures, the engine stays text-only.
    #[cfg(all(feature = "multimodal", feature = "cuda"))]
    let hf_config = &core.hf_config;
    #[cfg(all(feature = "multimodal", feature = "cuda"))]
    if let Some(processor) = crate::multimodal::resolve(
        &hf_config.architectures,
        &hf_config.extra,
        core.model_dir.as_deref(),
    ) {
        info!(
            "Multimodal config: arch={:?}, image_token_id={}, image_size={}, tokens_per_image={}",
            hf_config.architectures,
            processor.image_token_id,
            processor.image_size,
            processor.mm_tokens_per_image,
        );
        engine.set_mm_processor(Some(processor));
    }

    info!(
        "init engine (load model, create kv cache) took {:.2} seconds",
        init_start.elapsed().as_secs_f64()
    );

    // 11. Finish progress bar and return the stack.
    progress.finish();

    Ok(InitializedStack {
        engine: Arc::new(engine),
        model_name,
        max_model_len,
    })
}

/// Multi-node init flow ("mp" backend): leader creates engine+scheduler and
/// broadcasts scheduler output via TCP. Follower runs headless, receiving
/// commands via TCP and participating in NCCL collectives during forward.
///
/// - Leader (node_rank=0): Creates FerriteWorker, NCCL comm, cache init,
///   warmup, wraps in `MultiNodeExecutor`, returns `InitializedStack`.
/// - Follower (node_rank>0): Handled by [`initialize_and_run_follower`].
fn initialize_stack_multinode(
    config: &VllmConfig,
    model_name: String,
    init_start: Instant,
) -> Result<InitializedStack> {
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (config, model_name, init_start);
        anyhow::bail!(
            "Multi-node TP requires the `nccl` feature; \
             rebuild with --features nccl"
        );
    }

    #[cfg(feature = "nccl")]
    {
        use vllm_executor::ferrite_worker::{FerriteWorker, FerriteWorkerConfig};
        use vllm_executor::multinode::MultiNodeExecutor;

        let tp_size = config.tensor_parallel_size;
        let node_rank = config.node_rank;
        let num_nodes = config.num_nodes;
        let is_pooling = config.runner == "pooling";

        if node_rank != 0 {
            anyhow::bail!(
                "initialize_stack_multinode called on follower (node_rank={}). \
                 Use initialize_and_run_follower instead.",
                node_rank
            );
        }

        info!(
            "Multi-node TP: leader node, tp_size={}, num_nodes={}, master={}:{}",
            tp_size, num_nodes, config.master_addr, config.master_port
        );

        // Step 1: Exchange NCCL unique ID via TCP store.
        let nccl_id_bytes = vllm_cuda::tcp_store::exchange_nccl_id(
            0,
            tp_size,
            &config.master_addr,
            config.master_port,
        )
        .context("failed to exchange NCCL ID via TCP store")?;
        let nccl_id = vllm_cuda::NcclId::from_raw(nccl_id_bytes);
        info!("Leader: NCCL ID exchanged");

        // Step 2: Create FerriteWorker for this node's GPU (rank 0).
        let cuda_config = FerriteWorkerConfig {
            model_path: config.model.clone(),
            dtype: config.dtype.clone(),
            hf_token: config.hf_token.clone(),
            block_size: config.block_size,
            device_id: 0,
            enforce_eager: config.enforce_eager,
            max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
            cuda_graph_sizes: config
                .cuda_graph_config
                .as_ref()
                .map(|c| c.capture_sizes.clone())
                .unwrap_or_default(),
            cublas_autotune: config.cublas_autotune,
            gpu_memory_utilization: config.gpu_memory_utilization,
            pooling_strategy: config.pooling_strategy.clone(),
            is_pooling,
            tp_rank: 0,
            tp_world_size: tp_size,
            pp_rank: 0,
            pp_size: 1,
            gguf_file: config.gguf_file.clone(),
            lora_adapter: config.lora_adapter.clone(),
            kv_cache_dtype: config.kv_cache_dtype.clone(),
            calculate_kv_scales: config.calculate_kv_scales,
            cuda_graph_mode: config
                .cuda_graph_mode
                .parse()
                .unwrap_or(CudaGraphMode::Auto),
            eos_token_ids: vec![],
            max_model_len: config.max_model_len,
            draft_model_path: spec_decode_draft_model_path(config),
            draft_model_dtype: config.draft_model_dtype.clone(),
        };

        let mut worker = FerriteWorker::new(cuda_config);
        worker
            .init_device()
            .context("failed to initialize CUDA device")?;
        worker.load_model().context("failed to load model")?;

        let hf_config = worker
            .hf_config()
            .context("model config not available after load")?
            .clone();
        let model_dir = worker.model_dir().map(|p| p.to_path_buf());
        let dtype_elem_bytes = worker.resolved_dtype_elem_bytes();

        // Step 3: Create NCCL communicator (collective — all ranks participate).
        let device = worker.device_ref().expect("device not initialized");
        unsafe {
            vllm_cuda::driver::ctx_set_current(device.ctx).unwrap();
        }
        let nccl_group = vllm_cuda::NcclGroup::new(0, tp_size, nccl_id, device.compute_stream)
            .context("NCCL comm init failed")?;
        worker.set_tp_group(std::sync::Arc::new(nccl_group));
        info!("Leader: NCCL communicator created");

        // Step 4: Profile available memory.
        let available_memory = worker
            .determine_available_memory()
            .context("failed to determine available memory")?;

        // Step 5: All-reduce MIN across ranks via TCP store.
        let min_memory = vllm_cuda::tcp_store::allreduce_min(
            0,
            tp_size,
            available_memory,
            &config.master_addr,
            config.master_port,
        )
        .context("failed to allreduce memory")?;

        let max_model_len = config
            .max_model_len
            .or(hf_config.max_position_embeddings)
            .unwrap_or(4096);

        info!(
            "Leader: available_memory={:.1} GB, min_across_ranks={:.1} GB",
            available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            min_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        // Step 6: Compute block count.
        let num_gpu_blocks = compute_num_blocks(
            min_memory,
            config.block_size,
            &hf_config,
            dtype_elem_bytes,
            config.gpu_memory_utilization,
            &config.kv_cache_dtype,
        );

        info!(
            "Leader: num_gpu_blocks={}, kv_cache_tokens={}",
            num_gpu_blocks,
            num_gpu_blocks * config.block_size,
        );

        // Step 7: Establish TCP control channel (persistent connections).
        let mut channel = vllm_cuda::TcpControlChannel::establish(
            0,
            tp_size,
            &config.master_addr,
            config.master_port,
        )
        .context("failed to establish TCP control channel")?;
        info!("Leader: TCP control channel established");

        // Step 8: Initialize cache — broadcast command to followers first,
        // then run locally. Both sides participate in any NCCL collectives.
        {
            use vllm_executor::multinode::ControlMessage;
            let msg = ControlMessage::InitCache {
                num_gpu_blocks,
                num_cpu_blocks: 0,
            };
            let data = bincode::serialize(&msg).context("serialize InitCache")?;
            channel.broadcast(&data).context("broadcast InitCache")?;
        }
        let mut worker: Box<dyn Worker> = Box::new(worker);
        worker
            .initialize_cache(num_gpu_blocks, 0)
            .context("failed to initialize cache")?;

        // Step 9: Warm up — broadcast to followers, then run locally.
        // Both sides' forward passes hit NCCL collectives simultaneously.
        {
            use vllm_executor::multinode::ControlMessage;
            let msg = ControlMessage::Warmup;
            let data = bincode::serialize(&msg).context("serialize Warmup")?;
            channel.broadcast(&data).context("broadcast Warmup")?;
        }
        worker
            .compile_or_warm_up_model()
            .context("failed to compile or warm up model")?;

        // Step 10: Wrap in UniProcExecutor → MultiNodeExecutor.
        let executor = UniProcExecutor::new_pre_initialized(worker);
        let multi_executor = MultiNodeExecutor::new(Box::new(executor), channel);

        // Step 11: Build engine.
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

        let use_async_scheduling =
            (!config.disable_async_scheduling && !spec_decode_requires_sync(config));
        let enable_prefix_caching = config.enable_prefix_caching;
        let engine_config = EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
                max_num_seqs: config.max_num_seqs,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                async_scheduling: Some(use_async_scheduling),
                num_lookahead_tokens: if config.speculative_model.is_some() {
                    config.num_speculative_tokens
                } else {
                    0
                },
                ..Default::default()
            },
            max_model_len,
            num_gpu_blocks,
            block_size: config.block_size,
            engine_index: 0,
            async_scheduling: use_async_scheduling,
            use_spec_decode: config.speculative_model.is_some(),
            proposer_config: None,
            eos_token_ids,
            is_pooling: config.runner == "pooling",
            enable_prefix_caching,
        };

        let client: Box<dyn vllm_engine::core_client::EngineCoreClient + Send> =
            Box::new(InprocClient::new(engine_config, Box::new(multi_executor)));

        // Load tokenizer and build engine.
        let tokenizer = model_dir
            .as_ref()
            .and_then(|dir| try_load_tokenizer(dir).ok());

        let mut engine = if let Some(tok) = tokenizer {
            let tokenizer = Arc::new(tok);
            if let Some(ref dir) = model_dir {
                if let Some(ct) = resolve_chat_template(config.chat_template.as_deref(), dir) {
                    info!("Chat template loaded from tokenizer_config.json");
                    AsyncEngine::with_tokenizer_and_template(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                        Arc::new(ct),
                    )
                } else {
                    AsyncEngine::with_tokenizer(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                    )
                }
            } else {
                AsyncEngine::with_tokenizer(client, model_name.clone(), max_model_len, tokenizer)
            }
        } else {
            AsyncEngine::new(client, model_name.clone(), max_model_len)
        };

        if (!config.disable_async_scheduling && !spec_decode_requires_sync(config)) {
            engine.set_async_scheduling(true);
        }
        if config.runner == "pooling" {
            engine.set_is_pooling(true);
        }

        info!(
            "Leader: stack initialized in {:.1}s (multi-node, tp_size={})",
            init_start.elapsed().as_secs_f64(),
            tp_size,
        );

        Ok(InitializedStack {
            engine: Arc::new(engine),
            model_name,
            max_model_len,
        })
    }
}

/// Initialize a follower node and run the headless worker loop.
///
/// This function does NOT return (blocks forever in the headless loop)
/// until the leader sends a Shutdown command or the connection drops.
///
/// Called from the CLI when `node_rank > 0` and `num_nodes > 1`.
#[cfg(feature = "nccl")]
pub fn initialize_and_run_follower(config: &VllmConfig) -> Result<()> {
    use vllm_executor::ferrite_worker::{FerriteWorker, FerriteWorkerConfig};

    validate_speculative_decoding(config)?;

    let tp_size = config.tensor_parallel_size;
    let node_rank = config.node_rank;
    let is_pooling = config.runner == "pooling";

    info!(
        "Multi-node TP: follower node_rank={}, tp_size={}, master={}:{}",
        node_rank, tp_size, config.master_addr, config.master_port
    );

    // Step 1: Exchange NCCL unique ID via TCP store.
    let nccl_id_bytes = vllm_cuda::tcp_store::exchange_nccl_id(
        node_rank,
        tp_size,
        &config.master_addr,
        config.master_port,
    )
    .context("failed to exchange NCCL ID via TCP store")?;
    let nccl_id = vllm_cuda::NcclId::from_raw(nccl_id_bytes);
    info!("Follower {}: NCCL ID exchanged", node_rank);

    // Step 2: Create FerriteWorker for this node's GPU.
    let cuda_config = FerriteWorkerConfig {
        model_path: config.model.clone(),
        dtype: config.dtype.clone(),
        hf_token: config.hf_token.clone(),
        block_size: config.block_size,
        device_id: 0, // Each node has 1 GPU at device 0.
        enforce_eager: config.enforce_eager,
        max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
        cuda_graph_sizes: config
            .cuda_graph_config
            .as_ref()
            .map(|c| c.capture_sizes.clone())
            .unwrap_or_default(),
        cublas_autotune: config.cublas_autotune,
        gpu_memory_utilization: config.gpu_memory_utilization,
        pooling_strategy: config.pooling_strategy.clone(),
        is_pooling,
        tp_rank: node_rank,
        tp_world_size: tp_size,
        gguf_file: config.gguf_file.clone(),
        lora_adapter: config.lora_adapter.clone(),
        kv_cache_dtype: config.kv_cache_dtype.clone(),
        calculate_kv_scales: config.calculate_kv_scales,
        pp_rank: 0,
        pp_size: 1,
        cuda_graph_mode: config
            .cuda_graph_mode
            .parse()
            .unwrap_or(CudaGraphMode::Auto),
        eos_token_ids: vec![],
        max_model_len: config.max_model_len,
        draft_model_path: spec_decode_draft_model_path(config),
        draft_model_dtype: config.draft_model_dtype.clone(),
    };

    let mut worker = FerriteWorker::new(cuda_config);
    worker
        .init_device()
        .context("failed to initialize CUDA device")?;
    worker.load_model().context("failed to load model")?;

    // Step 3: Create NCCL communicator (collective — all ranks participate).
    let device = worker.device_ref().expect("device not initialized");
    unsafe {
        vllm_cuda::driver::ctx_set_current(device.ctx).unwrap();
    }
    let nccl_group = vllm_cuda::NcclGroup::new(node_rank, tp_size, nccl_id, device.compute_stream)
        .context("NCCL comm init failed")?;
    worker.set_tp_group(std::sync::Arc::new(nccl_group));
    info!("Follower {}: NCCL communicator created", node_rank);

    // Step 4: Profile available memory.
    let available_memory = worker
        .determine_available_memory()
        .context("failed to determine available memory")?;

    // Step 5: All-reduce MIN across ranks via TCP store.
    let min_memory = vllm_cuda::tcp_store::allreduce_min(
        node_rank,
        tp_size,
        available_memory,
        &config.master_addr,
        config.master_port,
    )
    .context("failed to allreduce memory")?;

    info!(
        "Follower {}: available_memory={:.1} GB, min_across_ranks={:.1} GB",
        node_rank,
        available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        min_memory as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    // Step 6: Establish TCP control channel.
    let channel = vllm_cuda::TcpControlChannel::establish(
        node_rank,
        tp_size,
        &config.master_addr,
        config.master_port,
    )
    .context("failed to establish TCP control channel")?;
    info!("Follower {}: TCP control channel established", node_rank);

    // Step 7: Wrap in UniProcExecutor and enter headless loop.
    // The headless loop receives InitCache, Warmup, ExecuteModel, and Shutdown
    // commands from the leader via TCP. NCCL collectives in the forward pass
    // synchronize with the leader automatically.
    let executor = UniProcExecutor::new_pre_initialized(Box::new(worker));

    info!("Follower {}: entering headless loop", node_rank,);
    crate::headless::run_headless(executor, channel)?;

    info!(
        "Follower {}: headless loop exited, shutting down",
        node_rank
    );
    Ok(())
}

/// Multi-GPU init flow: creates N workers (one per GPU rank), inits NCCL,
/// and wraps them in a ThreadPoolExecutor.
///
/// Each worker loads the full model weights and keeps only its shard
/// Initialize full stack for TP+PP or PP-only (pipeline parallelism).
///
/// Creates `tp_size * pp_size` workers, each with a TP NCCL comm and a PP
/// NCCL comm. Workers load only their PP stage's layers.
fn initialize_stack_tp_pp(
    config: &VllmConfig,
    model_name: String,
    init_start: Instant,
) -> Result<InitializedStack> {
    let _ = (config, model_name, init_start);
    anyhow::bail!(
        "Pipeline parallelism is not supported on the ferrite-only forwards \
         (the previous PP path lived in `vllm_cuda::model::*` and has been \
         removed). Run with `--pipeline-parallel-size 1`."
    );
    #[allow(unreachable_code)]
    {
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (config, model_name, init_start);
        anyhow::bail!(
            "Pipeline parallelism requires the `nccl` feature; \
             rebuild with --features nccl"
        );
    }

    #[cfg(feature = "nccl")]
    {
        use vllm_executor::ferrite_worker::{FerriteWorker, FerriteWorkerConfig};
        use vllm_executor::parallel::ResolvedParallelConfig;
        use vllm_executor::threadpool::ThreadPoolExecutor;

        let tp_size = config.tensor_parallel_size;
        let pp_size = config.pipeline_parallel_size;
        let world_size = tp_size * pp_size;
        let is_pooling = config.runner == "pooling";

        info!(
            "Pipeline parallelism: {} PP stages × {} TP ranks = {} GPUs",
            pp_size, tp_size, world_size
        );

        // Generate NCCL unique IDs:
        // - One TP NcclId per PP stage (pp_size total — ranks in same stage share it).
        // - One PP NcclId per TP position (tp_size total — ranks with same tp_rank share it).
        let tp_nccl_ids: Vec<vllm_cuda::NcclId> = (0..pp_size)
            .map(|_| vllm_cuda::NcclId::new().context("failed to generate TP NCCL ID"))
            .collect::<Result<Vec<_>>>()?;
        let pp_nccl_ids: Vec<vllm_cuda::NcclId> = (0..tp_size)
            .map(|_| vllm_cuda::NcclId::new().context("failed to generate PP NCCL ID"))
            .collect::<Result<Vec<_>>>()?;

        // Build per-rank configs.
        // Rank layout: global_rank = pp_rank * tp_size + tp_rank.
        let worker_configs: Vec<FerriteWorkerConfig> = (0..world_size)
            .map(|global_rank| {
                let tp_rank = global_rank % tp_size;
                let pp_rank = global_rank / tp_size;
                FerriteWorkerConfig {
                    model_path: config.model.clone(),
                    dtype: config.dtype.clone(),
                    hf_token: config.hf_token.clone(),
                    block_size: config.block_size,
                    device_id: global_rank as i32,
                    enforce_eager: config.enforce_eager,
                    max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
                    cuda_graph_sizes: config
                        .cuda_graph_config
                        .as_ref()
                        .map(|c| c.capture_sizes.clone())
                        .unwrap_or_default(),
                    cublas_autotune: config.cublas_autotune,
                    gpu_memory_utilization: config.gpu_memory_utilization,
                    pooling_strategy: config.pooling_strategy.clone(),
                    is_pooling,
                    tp_rank,
                    tp_world_size: tp_size,
                    pp_rank,
                    pp_size,
                    gguf_file: config.gguf_file.clone(),
                    lora_adapter: config.lora_adapter.clone(),
                    kv_cache_dtype: config.kv_cache_dtype.clone(),
                    calculate_kv_scales: config.calculate_kv_scales,
                    cuda_graph_mode: config
                        .cuda_graph_mode
                        .parse()
                        .unwrap_or(CudaGraphMode::Auto),
                    eos_token_ids: vec![],
                    max_model_len: config.max_model_len,
                    draft_model_path: spec_decode_draft_model_path(config),
                    draft_model_dtype: config.draft_model_dtype.clone(),
                }
            })
            .collect();

        // Download barrier: rank 0 downloads first, others wait.
        let download_barrier = std::sync::Arc::new(std::sync::Barrier::new(world_size));

        // Phase 1: Spawn one thread per GPU — init device + load model (PP-aware).
        let handles: Vec<_> = worker_configs
            .into_iter()
            .enumerate()
            .map(|(global_rank, cfg)| {
                let barrier = download_barrier.clone();
                std::thread::spawn(move || -> Result<FerriteWorker> {
                    let mut worker = FerriteWorker::new(cfg);
                    worker.init_device().context("init_device failed")?;

                    // Rank 0 loads first (downloads model files to cache).
                    if global_rank == 0 {
                        worker.load_model().context("load_model failed")?;
                        barrier.wait();
                    } else {
                        barrier.wait();
                        worker.load_model().context("load_model failed")?;
                    }

                    Ok(worker)
                })
            })
            .collect();

        // Collect workers.
        let mut ferrite_workers: Vec<FerriteWorker> = Vec::with_capacity(world_size);
        let mut hf_config = None;
        let mut model_dir = None;
        let mut dtype_elem_bytes: usize = 2; // BF16 default

        for (rank, handle) in handles.into_iter().enumerate() {
            let worker = handle
                .join()
                .map_err(|_| anyhow::anyhow!("worker thread {rank} panicked"))?
                .with_context(|| format!("worker {rank} init failed"))?;

            if rank == 0 {
                hf_config = worker.hf_config().cloned();
                model_dir = worker.model_dir().map(|p| p.to_path_buf());
                dtype_elem_bytes = worker.resolved_dtype_elem_bytes();
            }
            ferrite_workers.push(worker);
        }

        let hf_config = hf_config.context("model config not available after load")?;

        let max_model_len = config
            .max_model_len
            .or(hf_config.max_position_embeddings)
            .unwrap_or(4096);
        let num_layers = hf_config.num_hidden_layers.unwrap_or(1);

        info!(
            "Model: {}, max_model_len={}, num_layers={}, tp={}, pp={}",
            model_name, max_model_len, num_layers, tp_size, pp_size
        );

        // Phase 2: Create NCCL comms (TP + PP) + profile memory + warmup.
        // All NCCL init calls require ranks in the same group to participate
        // simultaneously, so we use scoped threads.
        //
        // We create TP comms first (all ranks in each TP group sync), then PP
        // comms (all ranks in each PP group sync). This ordering ensures no
        // deadlock since all ranks follow the same order.
        let init_results: Vec<Result<(usize, FerriteWorker)>> = std::thread::scope(|s| {
            let handles: Vec<_> = ferrite_workers
                .into_iter()
                .enumerate()
                .map(|(global_rank, mut worker)| {
                    let tp_nccl_ids = &tp_nccl_ids;
                    let pp_nccl_ids = &pp_nccl_ids;
                    s.spawn(move || -> Result<(usize, FerriteWorker)> {
                        let tp_rank = global_rank % tp_size;
                        let pp_rank = global_rank / tp_size;

                        // Set CUDA context.
                        let device = worker.device_ref().expect("device not initialized");
                        unsafe {
                            vllm_cuda::driver::ctx_set_current(device.ctx).unwrap();
                        }

                        // Create TP NCCL comm (ranks in the same PP stage).
                        if tp_size > 1 {
                            let tp_nccl_id = tp_nccl_ids[pp_rank];
                            let nccl_group = vllm_cuda::NcclGroup::new(
                                tp_rank,
                                tp_size,
                                tp_nccl_id,
                                device.compute_stream,
                            )
                            .with_context(|| {
                                format!("TP NCCL comm init failed for rank {global_rank}")
                            })?;
                            worker.set_tp_group(std::sync::Arc::new(nccl_group));
                        }

                        // Create PP NCCL comm (ranks with the same TP rank).
                        let pp_nccl_id = pp_nccl_ids[tp_rank];
                        let device = worker.device_ref().unwrap();
                        let pp_group = vllm_cuda::NcclGroup::new(
                            pp_rank,
                            pp_size,
                            pp_nccl_id,
                            device.compute_stream,
                        )
                        .with_context(|| {
                            format!("PP NCCL comm init failed for rank {global_rank}")
                        })?;
                        worker.set_pp_group(std::sync::Arc::new(pp_group));

                        // Allocate PP recv buffers on non-first stages.
                        worker.allocate_pp_recv_buffers();

                        // Profile activation memory with dummy forward.
                        let avail = worker.determine_available_memory().map_err(|e| {
                            anyhow::anyhow!("determine_available_memory rank {global_rank}: {e}")
                        })?;

                        Ok((avail, worker))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut workers: Vec<Box<dyn Worker>> = Vec::with_capacity(world_size);
        let mut min_avail = usize::MAX;
        for res in init_results {
            let (avail, worker) = res?;
            min_avail = min_avail.min(avail);
            workers.push(Box::new(worker));
        }

        // num_gpu_blocks = min across ALL workers (matches Python).
        let num_gpu_blocks = compute_num_blocks(
            min_avail,
            config.block_size,
            &hf_config,
            dtype_elem_bytes,
            config.gpu_memory_utilization,
            &config.kv_cache_dtype,
        );

        // Initialize cache: each worker allocates KV for its layer count only.
        for w in &mut workers {
            w.initialize_cache(num_gpu_blocks, 0)
                .context("initialize_cache")?;
        }

        info!(
            "TP+PP: min available memory across ranks: {:.1} GB, num_gpu_blocks={}",
            min_avail as f64 / (1024.0 * 1024.0 * 1024.0),
            num_gpu_blocks,
        );

        // Warm up (CUDA graph capture) concurrently — forwards use NCCL collectives.
        let warmup_results: Vec<Result<()>> = std::thread::scope(|s| {
            let handles: Vec<_> = workers
                .iter_mut()
                .enumerate()
                .map(|(rank, w)| {
                    s.spawn(move || {
                        w.compile_or_warm_up_model()
                            .with_context(|| format!("compile_or_warm_up_model rank {rank}"))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for res in warmup_results {
            res?;
        }

        // Wrap in ThreadPoolExecutor with TP+PP config.
        let parallel_config = ResolvedParallelConfig::tensor_pipeline_parallel(tp_size, pp_size, 0);
        let executor = ThreadPoolExecutor::new(workers, parallel_config);

        // Build engine (same as TP-only path).
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

        // PP: force sync scheduling — async scheduling with PP requires token broadcast
        // from last stage to non-last stages, which is not yet implemented.
        let use_async_scheduling = false;
        let enable_prefix_caching = config.enable_prefix_caching;
        let engine_config = EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
                max_num_seqs: config.max_num_seqs,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                async_scheduling: Some(use_async_scheduling),
                num_lookahead_tokens: if config.speculative_model.is_some() {
                    config.num_speculative_tokens
                } else {
                    0
                },
                use_pp: true,
                ..Default::default()
            },
            max_model_len,
            num_gpu_blocks,
            block_size: config.block_size,
            engine_index: 0,
            async_scheduling: use_async_scheduling,
            use_spec_decode: config.speculative_model.is_some(),
            proposer_config: None,
            eos_token_ids,
            is_pooling: config.runner == "pooling",
            enable_prefix_caching,
        };

        let client: Box<dyn vllm_engine::core_client::EngineCoreClient + Send> =
            Box::new(InprocClient::new(engine_config, Box::new(executor)));

        // Load tokenizer and build engine.
        let tokenizer = model_dir
            .as_ref()
            .and_then(|dir| try_load_tokenizer(dir).ok());

        let mut engine = if let Some(tok) = tokenizer {
            let tokenizer = Arc::new(tok);
            if let Some(ref dir) = model_dir {
                if let Some(ct) = resolve_chat_template(config.chat_template.as_deref(), dir) {
                    info!("Chat template loaded from tokenizer_config.json");
                    AsyncEngine::with_tokenizer_and_template(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                        Arc::new(ct),
                    )
                } else {
                    AsyncEngine::with_tokenizer(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                    )
                }
            } else {
                AsyncEngine::with_tokenizer(client, model_name.clone(), max_model_len, tokenizer)
            }
        } else {
            AsyncEngine::new(client, model_name.clone(), max_model_len)
        };

        if false {
            // PP: sync scheduling forced
            engine.set_async_scheduling(true);
        }
        if config.runner == "pooling" {
            engine.set_is_pooling(true);
        }

        let engine = Arc::new(engine);

        info!(
            "Stack initialized with TP={}, PP={} in {:.1}s",
            tp_size,
            pp_size,
            init_start.elapsed().as_secs_f64()
        );

        Ok(InitializedStack {
            engine,
            model_name,
            max_model_len,
        })
    }
    } // close `#[allow(unreachable_code)] {` wrapper
}

/// (via ColumnParallelLinear/RowParallelLinear sharding at load time).
///
/// TODO: Port to FerriteWorker.
fn initialize_stack_tp(
    config: &VllmConfig,
    model_name: String,
    init_start: Instant,
) -> Result<InitializedStack> {
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (config, model_name, init_start);
        anyhow::bail!(
            "Tensor parallelism requires the `nccl` feature; \
             rebuild with --features nccl"
        );
    }

    #[cfg(feature = "nccl")]
    {
        use vllm_executor::ferrite_worker::{FerriteWorker, FerriteWorkerConfig};
        use vllm_executor::parallel::ResolvedParallelConfig;
        use vllm_executor::threadpool::ThreadPoolExecutor;

        let tp_size = config.tensor_parallel_size;
        let is_pooling = config.runner == "pooling";

        info!("Tensor parallelism: {} GPUs", tp_size);

        // Generate NCCL unique ID on rank 0.
        let nccl_id = vllm_cuda::NcclId::new().context("failed to generate NCCL unique ID")?;

        // Build per-rank configs.
        let worker_configs: Vec<FerriteWorkerConfig> = (0..tp_size)
            .map(|rank| FerriteWorkerConfig {
                model_path: config.model.clone(),
                dtype: config.dtype.clone(),
                hf_token: config.hf_token.clone(),
                block_size: config.block_size,
                device_id: rank as i32,
                enforce_eager: config.enforce_eager,
                max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(2048),
                cuda_graph_sizes: config
                    .cuda_graph_config
                    .as_ref()
                    .map(|c| c.capture_sizes.clone())
                    .unwrap_or_default(),
                cublas_autotune: config.cublas_autotune,
                gpu_memory_utilization: config.gpu_memory_utilization,
                pooling_strategy: config.pooling_strategy.clone(),
                is_pooling,
                tp_rank: rank,
                tp_world_size: tp_size,
                pp_rank: 0,
                pp_size: 1,
                gguf_file: config.gguf_file.clone(),
                lora_adapter: config.lora_adapter.clone(),
                kv_cache_dtype: config.kv_cache_dtype.clone(),
                calculate_kv_scales: config.calculate_kv_scales,
                cuda_graph_mode: config
                    .cuda_graph_mode
                    .parse()
                    .unwrap_or(CudaGraphMode::Auto),
                eos_token_ids: vec![],
                max_model_len: config.max_model_len,
                draft_model_path: spec_decode_draft_model_path(config),
                draft_model_dtype: config.draft_model_dtype.clone(),
            })
            .collect();

        // Download barrier: rank 0 downloads first, others wait.
        let download_barrier = std::sync::Arc::new(std::sync::Barrier::new(tp_size));

        // Spawn one thread per GPU. Each: init device → load model → create NCCL comm.
        let handles: Vec<_> = worker_configs
            .into_iter()
            .enumerate()
            .map(|(local_rank, cfg)| {
                let barrier = download_barrier.clone();

                std::thread::spawn(move || -> Result<FerriteWorker> {
                    let mut worker = FerriteWorker::new(cfg);
                    worker.init_device().context("init_device failed")?;

                    // Rank 0 loads first (downloads model files to cache).
                    if local_rank == 0 {
                        worker.load_model().context("load_model failed")?;
                        barrier.wait();
                    } else {
                        barrier.wait();
                        worker.load_model().context("load_model failed")?;
                    }

                    Ok(worker)
                })
            })
            .collect();

        // Collect concrete FerriteWorkers (not yet boxed as dyn Worker).
        let mut ferrite_workers: Vec<FerriteWorker> = Vec::with_capacity(tp_size);
        let mut hf_config = None;
        let mut model_dir = None;
        let mut dtype_elem_bytes: usize = 2; // BF16 default

        for (rank, handle) in handles.into_iter().enumerate() {
            let worker = handle
                .join()
                .map_err(|_| anyhow::anyhow!("worker thread {rank} panicked"))?
                .with_context(|| format!("worker {rank} init failed"))?;

            if rank == 0 {
                hf_config = worker.hf_config().cloned();
                model_dir = worker.model_dir().map(|p| p.to_path_buf());
                dtype_elem_bytes = worker.resolved_dtype_elem_bytes();
            }
            ferrite_workers.push(worker);
        }

        let hf_config = hf_config.context("model config not available after load")?;

        let max_model_len = config
            .max_model_len
            .or(hf_config.max_position_embeddings)
            .unwrap_or(4096);
        let num_layers = hf_config.num_hidden_layers.unwrap_or(1);

        info!(
            "Model: {}, max_model_len={}, num_layers={}, tp={}",
            model_name, max_model_len, num_layers, tp_size
        );

        // Create NCCL comms + profile memory + warmup all on persistent threads.
        // NCCL collectives require all ranks to participate simultaneously, so all
        // TP operations (NCCL init, profiling forward, CUDA graph capture) run on
        // dedicated per-rank threads that persist throughout init.
        let init_results: Vec<Result<(usize, FerriteWorker)>> = std::thread::scope(|s| {
            let handles: Vec<_> = ferrite_workers
                .into_iter()
                .enumerate()
                .map(|(rank, mut worker)| {
                    s.spawn(move || -> Result<(usize, FerriteWorker)> {
                        // Create NCCL communicator (collective — all ranks participate).
                        let device = worker.device_ref().expect("device not initialized");
                        unsafe {
                            vllm_cuda::driver::ctx_set_current(device.ctx).unwrap();
                        }
                        let nccl_group = vllm_cuda::NcclGroup::new(
                            rank,
                            tp_size,
                            nccl_id,
                            device.compute_stream,
                        )
                        .context("NCCL comm init failed")?;
                        worker.set_tp_group(std::sync::Arc::new(nccl_group));

                        // Profile activation memory with dummy forward.
                        let avail = worker.determine_available_memory().map_err(|e| {
                            anyhow::anyhow!("determine_available_memory rank {rank}: {e}")
                        })?;

                        Ok((avail, worker))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut workers: Vec<Box<dyn Worker>> = Vec::with_capacity(tp_size);
        let mut min_avail = usize::MAX;
        for res in init_results {
            let (avail, worker) = res?;
            min_avail = min_avail.min(avail);
            workers.push(Box::new(worker));
        }

        let num_gpu_blocks = compute_num_blocks(
            min_avail,
            config.block_size,
            &hf_config,
            dtype_elem_bytes,
            config.gpu_memory_utilization,
            &config.kv_cache_dtype,
        );

        // initialize_cache doesn't run forwards, safe to call sequentially.
        for w in &mut workers {
            w.initialize_cache(num_gpu_blocks, 0)
                .context("initialize_cache")?;
        }

        info!(
            "TP: min available memory across ranks: {:.1} GB, num_gpu_blocks={}",
            min_avail as f64 / (1024.0 * 1024.0 * 1024.0),
            num_gpu_blocks,
        );

        // Warm up (CUDA graph capture) concurrently — forwards use NCCL collectives.
        let warmup_results: Vec<Result<()>> = std::thread::scope(|s| {
            let handles: Vec<_> = workers
                .iter_mut()
                .enumerate()
                .map(|(rank, w)| {
                    s.spawn(move || {
                        w.compile_or_warm_up_model()
                            .with_context(|| format!("compile_or_warm_up_model rank {rank}"))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for res in warmup_results {
            res?;
        }

        // Wrap in ThreadPoolExecutor.
        let parallel_config = ResolvedParallelConfig::tensor_parallel(tp_size, 0);
        let executor = ThreadPoolExecutor::new(workers, parallel_config);

        // Build engine.
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

        let use_async_scheduling =
            (!config.disable_async_scheduling && !spec_decode_requires_sync(config));
        let enable_prefix_caching = config.enable_prefix_caching;
        let engine_config = EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(2048),
                max_num_seqs: config.max_num_seqs,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                async_scheduling: Some(use_async_scheduling),
                num_lookahead_tokens: if config.speculative_model.is_some() {
                    config.num_speculative_tokens
                } else {
                    0
                },
                ..Default::default()
            },
            max_model_len,
            num_gpu_blocks,
            block_size: config.block_size,
            engine_index: 0,
            async_scheduling: use_async_scheduling,
            use_spec_decode: config.speculative_model.is_some(),
            proposer_config: None,
            eos_token_ids,
            is_pooling: config.runner == "pooling",
            enable_prefix_caching,
        };

        let client: Box<dyn vllm_engine::core_client::EngineCoreClient + Send> =
            Box::new(InprocClient::new(engine_config, Box::new(executor)));

        // Load tokenizer. Apply the same `additional_special_tokens`
        // patch as the single-rank path — Qwen2-VL chat templates
        // reference `<|image_pad|>` / `<|video_pad|>` which live on
        // `tokenizer_config.json::additional_special_tokens` but are
        // missing from `tokenizer.json` for some HF mirrors, so the
        // chat renderer errors out without this patch.
        let tokenizer = model_dir.as_ref().and_then(|dir| {
            try_load_tokenizer(dir).ok().map(|mut tok| {
                patch_additional_special_tokens(&mut tok, dir);
                tok
            })
        });

        let mut engine = if let Some(tok) = tokenizer {
            let tokenizer = Arc::new(tok);
            if let Some(ref dir) = model_dir {
                if let Some(ct) = resolve_chat_template(config.chat_template.as_deref(), dir) {
                    info!("Chat template loaded from tokenizer_config.json");
                    AsyncEngine::with_tokenizer_and_template(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                        Arc::new(ct),
                    )
                } else {
                    AsyncEngine::with_tokenizer(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                    )
                }
            } else {
                AsyncEngine::with_tokenizer(client, model_name.clone(), max_model_len, tokenizer)
            }
        } else {
            AsyncEngine::new(client, model_name.clone(), max_model_len)
        };

        if (!config.disable_async_scheduling && !spec_decode_requires_sync(config)) {
            engine.set_async_scheduling(true);
        }
        if config.runner == "pooling" {
            engine.set_is_pooling(true);
        }

        // Multimodal config — generic ferrite-vision inventory lookup.
        // See the single-rank `initialize_stack` path for the rationale;
        // this is the same call shape, threaded through the TP-init
        // engine instead.
        #[cfg(all(feature = "multimodal", feature = "cuda"))]
        if let Some(processor) = crate::multimodal::resolve(
            &hf_config.architectures,
            &hf_config.extra,
            model_dir.as_deref(),
        ) {
            info!(
                "Multimodal config: arch={:?}, image_token_id={}, image_size={}, tokens_per_image={}",
                hf_config.architectures,
                processor.image_token_id,
                processor.image_size,
                processor.mm_tokens_per_image,
            );
            engine.set_mm_processor(Some(processor));
        }

        let engine = Arc::new(engine);

        info!(
            "Stack initialized with TP={} in {:.1}s",
            tp_size,
            init_start.elapsed().as_secs_f64()
        );

        Ok(InitializedStack {
            engine,
            model_name,
            max_model_len,
        })
    }
}

/// Parsed distributed environment variables for external launcher mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalLauncherEnv {
    pub rank: usize,
    pub local_rank: usize,
    pub world_size: usize,
    pub master_addr: String,
    pub master_port: u16,
}

impl ExternalLauncherEnv {
    /// Parse distributed env vars (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT).
    ///
    /// Falls back to `config` values for MASTER_ADDR and MASTER_PORT if not set in env.
    pub fn from_env(config: &VllmConfig) -> Result<Self> {
        let rank: usize = std::env::var("RANK")
            .context("RANK env var not set (required for external_launcher)")?
            .parse()
            .context("RANK must be an integer")?;
        let local_rank: usize = std::env::var("LOCAL_RANK")
            .context("LOCAL_RANK env var not set (required for external_launcher)")?
            .parse()
            .context("LOCAL_RANK must be an integer")?;
        let world_size: usize = std::env::var("WORLD_SIZE")
            .context("WORLD_SIZE env var not set (required for external_launcher)")?
            .parse()
            .context("WORLD_SIZE must be an integer")?;
        let master_addr =
            std::env::var("MASTER_ADDR").unwrap_or_else(|_| config.master_addr.clone());
        let master_port: u16 = std::env::var("MASTER_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(config.master_port);

        // Validate consistency.
        if config.tensor_parallel_size > 1 && config.tensor_parallel_size != world_size {
            anyhow::bail!(
                "--tensor-parallel-size ({}) does not match WORLD_SIZE ({})",
                config.tensor_parallel_size,
                world_size
            );
        }

        Ok(Self {
            rank,
            local_rank,
            world_size,
            master_addr,
            master_port,
        })
    }
}

/// External launcher init path: one process per GPU, NCCL via TCP store.
///
/// Used when `--distributed-executor-backend external_launcher`. The job launcher
/// (torchrun, mpirun, SLURM) spawns N processes, each calling this function.
/// Each process:
/// 1. Reads RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from env.
/// 2. Sets CUDA device to LOCAL_RANK.
/// 3. Exchanges NCCL unique ID via TCP store (rank 0 serves, others connect).
/// 4. Creates a single FerriteWorker with TP sharding for this rank.
/// 5. Coordinates memory allocation via TCP all-reduce MIN.
/// 6. Wraps in UniProcExecutor (one worker per process).
/// 7. Builds AsyncEngine and returns the stack.
///
/// Each process runs its own HTTP server on a different port (set via --port).
fn initialize_stack_external(
    config: &VllmConfig,
    model_name: String,
    init_start: Instant,
) -> Result<InitializedStack> {
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (config, model_name, init_start);
        anyhow::bail!(
            "External launcher requires the `nccl` feature; \
             rebuild with --features nccl"
        );
    }

    #[cfg(feature = "nccl")]
    {
        use vllm_executor::ferrite_worker::{FerriteWorker, FerriteWorkerConfig};

        let env = ExternalLauncherEnv::from_env(config)
            .context("failed to read external launcher env vars")?;
        let rank = env.rank;
        let local_rank = env.local_rank;
        let world_size = env.world_size;
        let master_addr = env.master_addr;
        let master_port = env.master_port;

        let is_pooling = config.runner == "pooling";

        info!(
            "External launcher: rank={}, local_rank={}, world_size={}, master={}:{}",
            rank, local_rank, world_size, master_addr, master_port
        );

        // Step 1: Exchange NCCL unique ID via TCP store.
        let nccl_id_bytes =
            vllm_cuda::tcp_store::exchange_nccl_id(rank, world_size, &master_addr, master_port)
                .context("failed to exchange NCCL ID via TCP store")?;
        let nccl_id = vllm_cuda::NcclId::from_raw(nccl_id_bytes);
        info!("Rank {}: NCCL ID exchanged", rank);

        // Step 2: Create FerriteWorker for this rank's GPU.
        let cuda_config = FerriteWorkerConfig {
            model_path: config.model.clone(),
            dtype: config.dtype.clone(),
            hf_token: config.hf_token.clone(),
            block_size: config.block_size,
            device_id: local_rank as i32,
            enforce_eager: config.enforce_eager,
            max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(2048),
            cuda_graph_sizes: config
                .cuda_graph_config
                .as_ref()
                .map(|c| c.capture_sizes.clone())
                .unwrap_or_default(),
            cublas_autotune: config.cublas_autotune,
            gpu_memory_utilization: config.gpu_memory_utilization,
            pooling_strategy: config.pooling_strategy.clone(),
            is_pooling,
            tp_rank: rank,
            tp_world_size: world_size,
            pp_rank: 0,
            pp_size: 1,
            gguf_file: config.gguf_file.clone(),
            lora_adapter: config.lora_adapter.clone(),
            kv_cache_dtype: config.kv_cache_dtype.clone(),
            calculate_kv_scales: config.calculate_kv_scales,
            cuda_graph_mode: config
                .cuda_graph_mode
                .parse()
                .unwrap_or(CudaGraphMode::Auto),
            eos_token_ids: vec![],
            max_model_len: config.max_model_len,
            draft_model_path: spec_decode_draft_model_path(config),
            draft_model_dtype: config.draft_model_dtype.clone(),
        };

        let mut worker = FerriteWorker::new(cuda_config);
        worker
            .init_device()
            .context("failed to initialize CUDA device")?;
        worker.load_model().context("failed to load model")?;

        let hf_config = worker
            .hf_config()
            .context("model config not available after load")?
            .clone();
        let model_dir = worker.model_dir().map(|p| p.to_path_buf());
        let dtype_elem_bytes = worker.resolved_dtype_elem_bytes();

        // Step 3: Create NCCL communicator.
        let device = worker.device_ref().expect("device not initialized");
        unsafe {
            vllm_cuda::driver::ctx_set_current(device.ctx).unwrap();
        }
        let nccl_group =
            vllm_cuda::NcclGroup::new(rank, world_size, nccl_id, device.compute_stream)
                .context("NCCL comm init failed")?;
        worker.set_tp_group(std::sync::Arc::new(nccl_group));
        info!("Rank {}: NCCL communicator created", rank);

        // Step 4: Profile available memory.
        let available_memory = worker
            .determine_available_memory()
            .context("failed to determine available memory")?;

        // Step 5: All-reduce MIN across ranks via TCP store.
        let min_memory = vllm_cuda::tcp_store::allreduce_min(
            rank,
            world_size,
            available_memory,
            &master_addr,
            master_port,
        )
        .context("failed to allreduce memory")?;

        let max_model_len = config
            .max_model_len
            .or(hf_config.max_position_embeddings)
            .unwrap_or(4096);

        info!(
            "Rank {}: available_memory={:.1} GB, min_across_ranks={:.1} GB",
            rank,
            available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            min_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        // Step 6: Compute block count and initialize cache.
        let num_gpu_blocks = compute_num_blocks(
            min_memory,
            config.block_size,
            &hf_config,
            dtype_elem_bytes,
            config.gpu_memory_utilization,
            &config.kv_cache_dtype,
        );

        let mut worker: Box<dyn Worker> = Box::new(worker);
        worker
            .initialize_cache(num_gpu_blocks, 0)
            .context("failed to initialize cache")?;

        info!(
            "Rank {}: num_gpu_blocks={}, kv_cache_tokens={}",
            rank,
            num_gpu_blocks,
            num_gpu_blocks * config.block_size,
        );

        // Step 7: Warm up / CUDA graph capture.
        worker
            .compile_or_warm_up_model()
            .context("failed to compile or warm up model")?;

        // Step 8: Wrap in UniProcExecutor (single worker per process).
        let executor = UniProcExecutor::new_pre_initialized(worker);

        // Step 9: Build engine.
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

        let use_async_scheduling =
            (!config.disable_async_scheduling && !spec_decode_requires_sync(config));
        let enable_prefix_caching = config.enable_prefix_caching;
        let engine_config = EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(2048),
                max_num_seqs: config.max_num_seqs,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                async_scheduling: Some(use_async_scheduling),
                num_lookahead_tokens: if config.speculative_model.is_some() {
                    config.num_speculative_tokens
                } else {
                    0
                },
                ..Default::default()
            },
            max_model_len,
            num_gpu_blocks,
            block_size: config.block_size,
            engine_index: 0,
            async_scheduling: use_async_scheduling,
            use_spec_decode: config.speculative_model.is_some(),
            proposer_config: None,
            eos_token_ids,
            is_pooling: config.runner == "pooling",
            enable_prefix_caching,
        };

        let client: Box<dyn vllm_engine::core_client::EngineCoreClient + Send> =
            Box::new(InprocClient::new(engine_config, Box::new(executor)));

        // Load tokenizer and build engine.
        let tokenizer = model_dir
            .as_ref()
            .and_then(|dir| try_load_tokenizer(dir).ok());

        let mut engine = if let Some(tok) = tokenizer {
            let tokenizer = Arc::new(tok);
            if let Some(ref dir) = model_dir {
                if let Some(ct) = resolve_chat_template(config.chat_template.as_deref(), dir) {
                    info!("Chat template loaded from tokenizer_config.json");
                    AsyncEngine::with_tokenizer_and_template(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                        Arc::new(ct),
                    )
                } else {
                    AsyncEngine::with_tokenizer(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                    )
                }
            } else {
                AsyncEngine::with_tokenizer(client, model_name.clone(), max_model_len, tokenizer)
            }
        } else {
            AsyncEngine::new(client, model_name.clone(), max_model_len)
        };

        if (!config.disable_async_scheduling && !spec_decode_requires_sync(config)) {
            engine.set_async_scheduling(true);
        }
        if config.runner == "pooling" {
            engine.set_is_pooling(true);
        }

        info!(
            "Rank {}: stack initialized in {:.1}s (external launcher, world_size={})",
            rank,
            init_start.elapsed().as_secs_f64(),
            world_size,
        );

        Ok(InitializedStack {
            engine: Arc::new(engine),
            model_name,
            max_model_len,
        })
    }
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

/// Mirror `transformers`' `AutoTokenizer` behavior: after the
/// `tokenizer.json::added_tokens` list is loaded, scan
/// `tokenizer_config.json::additional_special_tokens` (a string list)
/// and register any entries missing from the former. Required for
/// arches like Qwen2-VL / Qwen2.5-VL that declare
/// `<|image_pad|>` / `<|video_pad|>` / `<|vision_pad|>` only in
/// `tokenizer_config.json` — without this, the chat template's
/// `<|image_pad|>` emission subword-splits at encode time and
/// `expand_image_placeholders` finds zero occurrences. Idempotent;
/// no-op when `tokenizer_config.json` is absent or has no
/// `additional_special_tokens` field. Applied to every tokenizer
/// path (preloaded via ferrite_worker's bare HfTokenizer load AND
/// `try_load_tokenizer`'s fallback path) so configurations converge.
fn patch_additional_special_tokens(tok: &mut Tokenizer, model_dir: &Path) {
    let config_path = model_dir.join("tokenizer_config.json");
    let Ok(bytes) = std::fs::read(&config_path) else {
        return;
    };
    let Ok(cfg) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return;
    };
    let Some(arr) = cfg
        .get("additional_special_tokens")
        .and_then(|v| v.as_array())
    else {
        return;
    };
    let contents: Vec<String> = arr
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    if !contents.is_empty() {
        tok.register_additional_special_tokens(&contents);
    }
}

/// Resolve a chat template, honoring an explicit operator override
/// (`config.chat_template`) before falling back to auto-detection.
///
/// `override_str` is interpreted as a filesystem path if the value
/// names a readable file; otherwise treated as inline Jinja. This
/// covers `--chat-template /path/to/tokenizer_config.json` as well
/// as raw template strings.
pub(crate) fn resolve_chat_template(
    override_str: Option<&str>,
    model_dir: &Path,
) -> Option<ChatTemplate> {
    if let Some(s) = override_str {
        let trimmed = s.trim();
        let from_path = std::path::Path::new(trimmed);
        let raw: Option<String> = if from_path.is_file() {
            // Either a .jinja template or a tokenizer_config.json. If
            // the file parses as JSON with a `chat_template` field,
            // use that; otherwise treat the file body as raw Jinja.
            std::fs::read_to_string(from_path).ok().map(|body| {
                match serde_json::from_str::<serde_json::Value>(&body) {
                    Ok(v) => v
                        .get("chat_template")
                        .and_then(|x| x.as_str().map(String::from))
                        .unwrap_or(body),
                    Err(_) => body,
                }
            })
        } else {
            Some(trimmed.to_string())
        };
        if let Some(template_str) = raw {
            match ChatTemplate::new(template_str) {
                Ok(mut tpl) => {
                    // GGUFs almost always lack bos/eos at the
                    // template-string level — the original
                    // tokenizer_config.json renders them via
                    // `{{- bos_token }}` / `{{ eos_token }}`. When the
                    // override is paired with a GGUF, pull bos/eos
                    // from the file's metadata (same path the
                    // auto-detect branch uses) so an operator-supplied
                    // template still gets the right special tokens.
                    if let Some(gguf_path) = gguf_in_or_under(model_dir)
                        && let Ok(gguf) = ferrite_gguf::GgufFile::open(&gguf_path)
                    {
                        tpl = apply_gguf_special_tokens(&gguf, tpl);
                    }
                    info!("Chat template loaded from --chat-template override");
                    return Some(tpl);
                }
                Err(e) => {
                    info!("Failed to parse --chat-template override: {e}");
                }
            }
        }
    }
    try_load_chat_template(model_dir)
}

/// Helper used by both `resolve_chat_template` and `try_load_chat_template`:
/// locate the .gguf file at, or directly inside, `model_dir`.
fn gguf_in_or_under(model_dir: &Path) -> Option<std::path::PathBuf> {
    if model_dir.is_file() && model_dir.extension().is_some_and(|e| e == "gguf") {
        Some(model_dir.to_path_buf())
    } else if model_dir.is_dir() {
        std::fs::read_dir(model_dir).ok().and_then(|mut it| {
            it.find_map(|entry| {
                let p = entry.ok()?.path();
                (p.extension().is_some_and(|e| e == "gguf")).then_some(p)
            })
        })
    } else {
        None
    }
}

/// Pull bos/eos token strings out of GGUF metadata and stamp them
/// onto a `ChatTemplate`. Does nothing if either id or the tokens
/// array is missing.
fn apply_gguf_special_tokens(gguf: &ferrite_gguf::GgufFile, mut tpl: ChatTemplate) -> ChatTemplate {
    if let Some(tokens_val) = gguf.metadata().get("tokenizer.ggml.tokens")
        && let Ok(tokens) = tokens_val.to_vec()
    {
        let resolve =
            |id: u32| -> Option<String> { tokens.get(id as usize)?.to_string().ok().cloned() };
        if let Some(bos_id) = gguf.get_metadata_u32("tokenizer.ggml.bos_token_id")
            && let Some(s) = resolve(bos_id)
        {
            tpl = tpl.with_bos_token(s);
        }
        if let Some(eos_id) = gguf.get_metadata_u32("tokenizer.ggml.eos_token_id")
            && let Some(s) = resolve(eos_id)
        {
            tpl = tpl.with_eos_token(s);
        }
    }
    tpl
}

/// Try to load a chat template. Sources tried, in order:
/// 1. `model_dir/tokenizer_config.json` (HF convention).
/// 2. `model_dir/chat_template.jinja` (some quantized repos).
/// 3. `tokenizer.chat_template` metadata in a `.gguf` file —
///    handles both `model_dir` being a `.gguf` file directly and
///    a directory containing one.
fn try_load_chat_template(model_dir: &Path) -> Option<ChatTemplate> {
    // GGUF: read from metadata. Try direct file, then look for any
    // .gguf file in the directory.
    if let Some(gguf_path) = gguf_in_or_under(model_dir)
        && let Ok(gguf) = ferrite_gguf::GgufFile::open(&gguf_path)
        && let Some(template_str) = ferrite_gguf::gguf_chat_template(&gguf)
    {
        match ChatTemplate::new(template_str) {
            Ok(tpl) => {
                let tpl = apply_gguf_special_tokens(&gguf, tpl);
                info!("Chat template loaded from GGUF metadata");
                return Some(tpl);
            }
            Err(e) => {
                info!("Failed to parse GGUF chat_template: {e}");
            }
        }
    }

    // Bare .gguf file path with no template in metadata: fall through to
    // checking the parent directory for tokenizer_config.json /
    // chat_template.jinja (some HF GGUF repos ship those alongside).
    let probe_dir: std::borrow::Cow<'_, Path> = if model_dir.is_file() {
        std::borrow::Cow::Owned(model_dir.parent().unwrap_or(model_dir).to_path_buf())
    } else {
        std::borrow::Cow::Borrowed(model_dir)
    };
    let model_dir = probe_dir.as_ref();

    let config_path = model_dir.join("tokenizer_config.json");
    match ChatTemplate::from_tokenizer_config(&config_path) {
        Ok(Some(tpl)) => Some(tpl),
        Ok(None) => {
            // Fallback: some models (e.g. AWQ quantized) store the template in a
            // separate Jinja file instead of embedding it in tokenizer_config.json.
            let jinja_path = model_dir.join("chat_template.jinja");
            if jinja_path.exists() {
                match std::fs::read_to_string(&jinja_path) {
                    Ok(template_str) => match ChatTemplate::new(template_str) {
                        Ok(tpl) => {
                            info!("Chat template loaded from chat_template.jinja");
                            Some(tpl)
                        }
                        Err(e) => {
                            info!("Failed to parse chat_template.jinja: {e}");
                            None
                        }
                    },
                    Err(e) => {
                        info!("Failed to read chat_template.jinja: {e}");
                        None
                    }
                }
            } else {
                None
            }
        }
        Err(e) => {
            info!("Failed to parse chat template: {e}");
            None
        }
    }
}

/// Compute the number of KV cache blocks from available memory.
///
/// `available_bytes` is the KV cache budget in bytes, already accounting for
/// `gpu_memory_utilization`. This matches Python vLLM where
/// `determine_available_memory` returns `total * util - non_kv_cache` and
/// the block count is simply `available / bytes_per_block`.
fn compute_num_blocks(
    available_bytes: usize,
    block_size: usize,
    hf_config: &HfModelConfig,
    dtype_elem_bytes: usize,
    _gpu_memory_utilization: f64,
    kv_cache_dtype: &str,
) -> usize {
    let num_layers = hf_config.num_hidden_layers.unwrap_or(1);
    let num_kv_heads = hf_config.num_kv_heads().unwrap_or(0);
    let head_dim = hf_config.head_dim().unwrap_or(0);

    // Each block holds block_size tokens of KV for all layers.
    // KV per token per layer = 2 * num_kv_heads * head_dim * sizeof(dtype)
    // FP8 KV cache stores 1 byte/element instead of 2 (BF16), doubling capacity.
    let elem_bytes = if kv_cache_dtype == "fp8_e4m3" || kv_cache_dtype == "fp8" {
        1
    } else {
        dtype_elem_bytes
    };
    let bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * elem_bytes;
    let bytes_per_block = block_size * num_layers * bytes_per_token_per_layer;

    if bytes_per_block == 0 {
        return 1024; // Fallback.
    }

    let num_blocks = available_bytes / bytes_per_block;

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
        let blocks = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, 4, 0.9, "auto");
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
        let blocks_f32 = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, 4, 0.9, "auto");
        let blocks_f16 = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, 2, 0.9, "auto");
        assert!(blocks_f16 > blocks_f32);
        // F16 should give approximately 2x the blocks.
        assert!((blocks_f16 as f64 / blocks_f32 as f64 - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_compute_num_blocks_zero_dim() {
        let config = HfModelConfig::default();
        let blocks = compute_num_blocks(1024, 16, &config, 4, 0.9, "auto");
        // Should fall back to 1024.
        assert_eq!(blocks, 1024);
    }

    #[test]
    fn test_compute_num_blocks_proportional_to_memory() {
        // compute_num_blocks no longer applies utilization internally —
        // the caller is responsible for providing the final KV cache budget.
        // Verify that halving the input memory halves the blocks.
        let config = HfModelConfig {
            num_hidden_layers: Some(32),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            hidden_size: Some(4096),
            ..Default::default()
        };
        let blocks_full = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, 2, 0.9, "auto");
        let blocks_half = compute_num_blocks(2 * 1024 * 1024 * 1024, 16, &config, 2, 0.9, "auto");
        let ratio = blocks_half as f64 / blocks_full as f64;
        assert!((ratio - 0.5).abs() < 0.01, "ratio was {ratio}");
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
        assert!(config.max_num_batched_tokens.is_none());
    }

    #[test]
    fn test_max_num_batched_tokens_default_none() {
        let config = VllmConfig::default();
        assert!(
            config.max_num_batched_tokens.is_none(),
            "default should be None (auto)"
        );
    }

    #[test]
    fn test_max_num_batched_tokens_explicit_override() {
        let config = VllmConfig {
            max_num_batched_tokens: Some(4096),
            ..VllmConfig::default()
        };
        assert_eq!(config.max_num_batched_tokens, Some(4096));
    }

    #[test]
    fn test_vllm_config_default_distributed_backend() {
        let config = VllmConfig::default();
        assert_eq!(config.distributed_executor_backend, "auto");
    }

    #[test]
    fn test_vllm_config_external_launcher() {
        let config = VllmConfig {
            distributed_executor_backend: "external_launcher".to_string(),
            ..VllmConfig::default()
        };
        assert_eq!(config.distributed_executor_backend, "external_launcher");
    }

    // -- ExternalLauncherEnv tests --
    // Note: these tests manipulate env vars, which is inherently global state.
    // We use a mutex to serialize them and restore original values.

    use std::sync::Mutex;
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    /// Helper: set env vars, run closure, restore originals.
    fn with_env_vars<F, R>(vars: &[(&str, Option<&str>)], f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _lock = ENV_MUTEX.lock().unwrap();
        let mut originals = Vec::new();
        for &(key, val) in vars {
            originals.push((key, std::env::var(key).ok()));
            // SAFETY: we hold ENV_MUTEX, serializing all env var access in tests.
            unsafe {
                match val {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
        let result = f();
        for (key, original) in originals {
            // SAFETY: same mutex guard still held.
            unsafe {
                match original {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
        result
    }

    #[test]
    fn test_external_launcher_env_all_vars_set() {
        with_env_vars(
            &[
                ("RANK", Some("1")),
                ("LOCAL_RANK", Some("1")),
                ("WORLD_SIZE", Some("4")),
                ("MASTER_ADDR", Some("10.0.0.1")),
                ("MASTER_PORT", Some("29600")),
            ],
            || {
                let config = VllmConfig::default();
                let env = ExternalLauncherEnv::from_env(&config).unwrap();
                assert_eq!(env.rank, 1);
                assert_eq!(env.local_rank, 1);
                assert_eq!(env.world_size, 4);
                assert_eq!(env.master_addr, "10.0.0.1");
                assert_eq!(env.master_port, 29600);
            },
        );
    }

    #[test]
    fn test_external_launcher_env_fallback_master_addr() {
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("2")),
                ("MASTER_ADDR", None),
                ("MASTER_PORT", None),
            ],
            || {
                let config = VllmConfig {
                    master_addr: "my-host".to_string(),
                    master_port: 12345,
                    ..VllmConfig::default()
                };
                let env = ExternalLauncherEnv::from_env(&config).unwrap();
                assert_eq!(env.master_addr, "my-host");
                assert_eq!(env.master_port, 12345);
            },
        );
    }

    #[test]
    fn test_external_launcher_env_missing_rank() {
        with_env_vars(
            &[
                ("RANK", None),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("2")),
            ],
            || {
                let config = VllmConfig::default();
                let err = ExternalLauncherEnv::from_env(&config).unwrap_err();
                assert!(
                    err.to_string().contains("RANK"),
                    "error should mention RANK: {}",
                    err
                );
            },
        );
    }

    #[test]
    fn test_external_launcher_env_missing_local_rank() {
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", None),
                ("WORLD_SIZE", Some("2")),
            ],
            || {
                let config = VllmConfig::default();
                let err = ExternalLauncherEnv::from_env(&config).unwrap_err();
                assert!(
                    err.to_string().contains("LOCAL_RANK"),
                    "error should mention LOCAL_RANK: {}",
                    err
                );
            },
        );
    }

    #[test]
    fn test_external_launcher_env_missing_world_size() {
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", None),
            ],
            || {
                let config = VllmConfig::default();
                let err = ExternalLauncherEnv::from_env(&config).unwrap_err();
                assert!(
                    err.to_string().contains("WORLD_SIZE"),
                    "error should mention WORLD_SIZE: {}",
                    err
                );
            },
        );
    }

    #[test]
    fn test_external_launcher_env_invalid_rank() {
        with_env_vars(
            &[
                ("RANK", Some("not_a_number")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("2")),
            ],
            || {
                let config = VllmConfig::default();
                let err = ExternalLauncherEnv::from_env(&config).unwrap_err();
                assert!(
                    err.to_string().contains("integer"),
                    "error should mention integer: {}",
                    err
                );
            },
        );
    }

    #[test]
    fn test_external_launcher_env_tp_mismatch() {
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("4")),
                ("MASTER_ADDR", Some("127.0.0.1")),
                ("MASTER_PORT", Some("29500")),
            ],
            || {
                let config = VllmConfig {
                    tensor_parallel_size: 2, // mismatch with WORLD_SIZE=4
                    ..VllmConfig::default()
                };
                let err = ExternalLauncherEnv::from_env(&config).unwrap_err();
                assert!(
                    err.to_string().contains("does not match"),
                    "error should mention mismatch: {}",
                    err
                );
            },
        );
    }

    #[test]
    fn test_external_launcher_env_tp_size_one_no_mismatch() {
        // tp_size=1 (default) should NOT conflict with any WORLD_SIZE.
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("8")),
                ("MASTER_ADDR", Some("127.0.0.1")),
                ("MASTER_PORT", Some("29500")),
            ],
            || {
                let config = VllmConfig::default(); // tp_size=1
                let env = ExternalLauncherEnv::from_env(&config).unwrap();
                assert_eq!(env.world_size, 8);
            },
        );
    }

    #[test]
    fn test_external_launcher_env_tp_matches_world_size() {
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("4")),
                ("MASTER_ADDR", Some("127.0.0.1")),
                ("MASTER_PORT", Some("29500")),
            ],
            || {
                let config = VllmConfig {
                    tensor_parallel_size: 4, // matches WORLD_SIZE=4
                    ..VllmConfig::default()
                };
                let env = ExternalLauncherEnv::from_env(&config).unwrap();
                assert_eq!(env.world_size, 4);
            },
        );
    }

    #[test]
    fn test_external_launcher_env_rank0() {
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("1")),
            ],
            || {
                let config = VllmConfig::default();
                let env = ExternalLauncherEnv::from_env(&config).unwrap();
                assert_eq!(env.rank, 0);
                assert_eq!(env.local_rank, 0);
                assert_eq!(env.world_size, 1);
            },
        );
    }
}
