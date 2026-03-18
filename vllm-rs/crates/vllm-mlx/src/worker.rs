// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! MLX worker implementing the `Worker` trait from `vllm-executor`.
//!
//! All model ops are lazy: the forward pass builds a compute graph, and a
//! single `eval()` call materializes the entire graph as minimal Metal
//! command buffers.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};
use tracing::{debug, info, warn};

use vllm_common::SamplingParams;
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_engine::executor::ModelRunnerOutput;
use vllm_executor::error::{ExecutorError, ExecutorResult};
use vllm_executor::worker::Worker;
use vllm_model::weight::HfModelConfig;

use crate::cache::{self, BatchMlxLayerKvCache, MlxKvCache};
use crate::models::{MlxModel, MlxModelRegistry};

/// State saved between execute_model calls for double-buffered async_eval.
///
/// After async_eval submits the current step, we store the unread GPU
/// sampling arrays and the pre-built output.  At the start of the NEXT
/// execute_model call, we read these arrays (instant — GPU finished during
/// the main thread's finalize + schedule), update token_buffers, and
/// return the stored output.
struct PendingStep {
    /// Pre-built output for the previous step (req_ids, sampled_token_ids
    /// are NOT yet populated — token_ids_placeholder is empty vecs).
    req_ids: Vec<String>,
    /// GPU greedy result array (unread).  `None` if no greedy requests.
    greedy_result: Option<Array>,
    /// Per-request index into greedy_result's flat output.
    greedy_mapping: Vec<(String, usize)>, // (req_id, batch_pos)
    /// GPU temperature result arrays (unread), one per temperature group.
    temp_results: Vec<(Vec<(String, usize)>, Array)>, // ([(req_id, batch_pos)], array)
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for an `MlxWorker`.
#[derive(Debug, Clone)]
pub struct MlxWorkerConfig {
    /// Path to a local model directory, or a HuggingFace model ID.
    pub model_path: String,

    /// Data type for model weights: "auto", "f32", "f16", "bf16".
    pub dtype: String,

    /// Optional HuggingFace token for gated models.
    pub hf_token: Option<String>,

    /// Optional cache directory for downloaded models.
    pub cache_dir: Option<String>,

    /// KV cache block size in tokens.
    pub block_size: usize,

    /// Optional path to a LoRA adapter directory (local path or HF repo ID).
    pub lora_adapter: Option<String>,
    /// Pooling strategy for embeddings: "auto", "last", "cls", "mean".
    /// "auto" detects from `1_Pooling/config.json`, defaults to "last".
    pub pooling_strategy: String,
    /// Whether the engine is in pooling mode.
    /// In pooling mode, `execute_model` returns embedding vectors instead of sampled tokens.
    pub is_pooling: bool,
    /// Whether prefix caching (KV cache reuse) is enabled.
    pub enable_prefix_caching: bool,
}

impl MlxWorkerConfig {
    /// Parse the dtype string to an MLX Dtype.
    ///
    /// Returns `None` for "auto" — the caller must resolve from config.json.
    pub fn mlx_dtype(&self) -> ExecutorResult<Option<Dtype>> {
        match self.dtype.as_str() {
            "auto" => Ok(None),
            "f32" | "float32" => Ok(Some(Dtype::Float32)),
            "f16" | "float16" => Ok(Some(Dtype::Float16)),
            "bf16" | "bfloat16" => Ok(Some(Dtype::Bfloat16)),
            other => Err(ExecutorError::Config(format!("unsupported dtype: {other}"))),
        }
    }
}

// ---------------------------------------------------------------------------
// MlxWorker
// ---------------------------------------------------------------------------

/// Maximum number of KV caches retained in the prefix cache pool.
const PREFIX_CACHE_POOL_MAX: usize = 32;

/// Persistent batched KV caches for decode.
///
/// Persists across steps when the batch composition is stable (same request IDs
/// in the same order). Rebuilt when requests join or leave.
struct BatchedDecodeCache {
    /// Ordered request IDs in this batch.
    req_ids: Vec<String>,
    /// One batched KV cache per model layer.
    layer_caches: Vec<BatchMlxLayerKvCache>,
    /// Per layer, per request left-padding offset.
    left_padding: Vec<Vec<usize>>,
}

/// Hash the block-aligned prefix of a prompt (same logic as `SimpleBlockTracker::hash_block`).
///
/// Only full blocks are hashed — trailing partial blocks are ignored so that
/// the hash matches the scheduler's prefix lookup.
fn hash_prefix(prompt: &[u32], block_size: usize) -> u64 {
    let num_full_blocks = prompt.len() / block_size;
    let prefix_len = num_full_blocks * block_size;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    prompt[..prefix_len].hash(&mut hasher);
    hasher.finish()
}

/// A worker backed by MLX for Apple Silicon GPU inference.
///
/// Uses lazy evaluation: forward passes build compute graphs that are
/// materialized with a single `eval()` call, fusing hundreds of ops into
/// minimal Metal command buffers.
pub struct MlxWorker {
    config: MlxWorkerConfig,
    model: Option<Box<dyn MlxModel>>,
    /// Resolved local path to the model directory.
    model_dir: Option<PathBuf>,
    /// Parsed HuggingFace config.json.
    hf_config: Option<HfModelConfig>,
    /// Resolved model dtype.
    resolved_dtype: Option<Dtype>,
    /// KV cache block counts.
    num_gpu_blocks: usize,
    is_shutdown: bool,

    /// Per-request token buffer: req_id → all token IDs (prompt + generated).
    token_buffers: HashMap<String, Vec<u32>>,
    /// Per-request sampling params.
    sampling_params_map: HashMap<String, SamplingParams>,
    /// Per-request KV cache (simple contiguous, no paging).
    kv_caches: HashMap<String, MlxKvCache>,
    /// Per-request recurrent state for hybrid models (e.g., Qwen3-Next GDN layers).
    recurrent_states: HashMap<String, super::models::MlxRecurrentState>,

    /// Per-request grammar guide state for constrained decoding.
    #[cfg(feature = "guided-decoding")]
    grammar_states: HashMap<String, vllm_model::grammar::GrammarGuide>,
    /// Parser factory for grammar-guided decoding (built once from tokenizer).
    #[cfg(feature = "guided-decoding")]
    grammar_factory: Option<std::sync::Arc<vllm_model::grammar::LlgParserFactory>>,

    /// Resolved pooling strategy for embeddings.
    pooling_strategy: vllm_model::embedding::PoolingStrategy,
    /// Per-request multimodal data, consumed on first forward (prefill).
    mm_data_map: HashMap<String, vllm_common::MultimodalData>,

    /// Pre-loaded tokenizer (parsed in parallel with weight loading).
    preloaded_tokenizer: Option<tokenizers::Tokenizer>,
    /// Resolved model architecture name (e.g. "LlamaForCausalLM").
    resolved_architecture: Option<String>,

    /// Whether the engine is in pooling mode.
    is_pooling: bool,

    /// Cached KV caches for prefix reuse: hash → kv_cache.
    kv_cache_pool: HashMap<u64, MlxKvCache>,
    /// Whether prefix caching is enabled.
    enable_prefix_caching: bool,
    /// Block size for prefix hashing (must match scheduler block size).
    prefix_block_size: usize,

    /// Persistent batched KV caches for decode. Persists across steps when
    /// the batch composition is stable. Rebuilt when requests join/leave.
    batched_decode_cache: Option<BatchedDecodeCache>,

    /// Double-buffer state: previous step's unread GPU sampling arrays and
    /// the pre-built ModelRunnerOutput.  On the next execute_model call, we
    /// read these arrays (instant — GPU already finished), update
    /// token_buffers, and return this output.  See `project_double_buffer_design`.
    pending_step: Option<PendingStep>,

    // Timing instrumentation.
    step_count: usize,
    prefill_count: usize,
    decode_count: usize,
    total_prefill_ms: f64,
    total_decode_ms: f64,
}

impl MlxWorker {
    /// Create a new MlxWorker from the given config.
    pub fn new(config: MlxWorkerConfig) -> Self {
        let is_pooling = config.is_pooling;
        let enable_prefix_caching = config.enable_prefix_caching;
        let prefix_block_size = config.block_size;
        Self {
            config,
            model: None,
            model_dir: None,
            hf_config: None,
            resolved_dtype: None,
            num_gpu_blocks: 0,
            is_shutdown: false,
            token_buffers: HashMap::new(),
            sampling_params_map: HashMap::new(),
            kv_caches: HashMap::new(),
            recurrent_states: HashMap::new(),
            #[cfg(feature = "guided-decoding")]
            grammar_states: HashMap::new(),
            #[cfg(feature = "guided-decoding")]
            grammar_factory: None,
            pooling_strategy: vllm_model::embedding::PoolingStrategy::Last,
            mm_data_map: HashMap::new(),
            preloaded_tokenizer: None,
            is_pooling,
            resolved_architecture: None,
            kv_cache_pool: HashMap::new(),
            enable_prefix_caching,
            prefix_block_size,
            batched_decode_cache: None,
            pending_step: None,
            step_count: 0,
            prefill_count: 0,
            decode_count: 0,
            total_prefill_ms: 0.0,
            total_decode_ms: 0.0,
        }
    }

    /// Get the resolved model directory.
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }

    /// Get the parsed HfModelConfig.
    pub fn hf_config(&self) -> Option<&HfModelConfig> {
        self.hf_config.as_ref()
    }

    /// Get the resolved model dtype.
    pub fn resolved_dtype(&self) -> Option<Dtype> {
        self.resolved_dtype
    }

    /// Total execute_model steps processed.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Number of prefill steps processed.
    pub fn prefill_count(&self) -> usize {
        self.prefill_count
    }

    /// Number of decode steps processed.
    pub fn decode_count(&self) -> usize {
        self.decode_count
    }

    /// Average prefill latency in milliseconds, or 0 if none.
    pub fn avg_prefill_ms(&self) -> f64 {
        if self.prefill_count == 0 {
            0.0
        } else {
            self.total_prefill_ms / self.prefill_count as f64
        }
    }

    /// Average decode latency in milliseconds, or 0 if none.
    pub fn avg_decode_ms(&self) -> f64 {
        if self.decode_count == 0 {
            0.0
        } else {
            self.total_decode_ms / self.decode_count as f64
        }
    }

    /// Resolve the pooling strategy from config or auto-detect.
    fn resolve_pooling_strategy(&mut self) {
        use vllm_model::embedding::{PoolingStrategy, detect_pooling_strategy};

        let strategy = match self.config.pooling_strategy.as_str() {
            "auto" => {
                let detected = self
                    .model_dir
                    .as_ref()
                    .and_then(|dir| detect_pooling_strategy(dir));
                if let Some(s) = detected {
                    info!("MlxWorker: pooling strategy auto-detected: {:?}", s);
                    s
                } else {
                    info!("MlxWorker: pooling strategy defaulting to Last (decoder model)");
                    PoolingStrategy::Last
                }
            }
            other => other.parse().unwrap_or_else(|_| {
                warn!(
                    "MlxWorker: unknown pooling strategy '{}', defaulting to Last",
                    other
                );
                PoolingStrategy::Last
            }),
        };
        self.pooling_strategy = strategy;
    }

    /// Build grammar parser factory on demand (lazy — deferred from startup).
    #[cfg(feature = "guided-decoding")]
    fn ensure_grammar_factory(&mut self) {
        if self.grammar_factory.is_some() {
            return;
        }
        let Some(model_dir) = &self.model_dir else {
            return;
        };
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            info!("MlxWorker: no tokenizer.json found, grammar-guided decoding unavailable");
            return;
        }
        let tokenizer_bytes = match std::fs::read(&tokenizer_path) {
            Ok(b) => b,
            Err(e) => {
                warn!("MlxWorker: failed to read tokenizer.json for grammar factory: {e}");
                return;
            }
        };

        match vllm_model::grammar::build_parser_factory(&tokenizer_bytes) {
            Ok(factory) => {
                info!("MlxWorker: grammar parser factory built");
                self.grammar_factory = Some(factory);
            }
            Err(e) => {
                warn!("MlxWorker: failed to build grammar parser factory: {e}");
            }
        }
    }

    /// Build an HF Hub API client from config.
    fn build_hf_api(&self) -> ExecutorResult<hf_hub::api::sync::Api> {
        let mut builder = hf_hub::api::sync::ApiBuilder::new();
        if let Some(token) = &self.config.hf_token {
            builder = builder.with_token(Some(token.clone()));
        }
        if let Some(cache_dir) = &self.config.cache_dir {
            builder = builder.with_cache_dir(PathBuf::from(cache_dir));
        }
        builder
            .build()
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to create HF API: {e}")))
    }

    /// Resolve a model path to a local directory.
    fn resolve_model_path(&self) -> ExecutorResult<PathBuf> {
        let path = Path::new(&self.config.model_path);
        if path.is_dir() {
            return Ok(path.to_path_buf());
        }

        // Treat as HuggingFace model ID — download via hf-hub.
        info!(
            "MlxWorker: downloading model from HuggingFace Hub: {}",
            self.config.model_path
        );
        let api = self.build_hf_api()?;
        let repo = api.model(self.config.model_path.clone());

        // Download config.json.
        let config_path = repo.get("config.json").map_err(|e| {
            ExecutorError::WorkerInit(format!(
                "failed to download config.json for {}: {e}",
                self.config.model_path
            ))
        })?;

        let model_dir = config_path
            .parent()
            .ok_or_else(|| {
                ExecutorError::WorkerInit("cannot determine model directory".to_string())
            })?
            .to_path_buf();

        // Download tokenizer (best effort).
        match repo.get("tokenizer.json") {
            Ok(p) => info!("Downloaded tokenizer.json to {}", p.display()),
            Err(e) => warn!("Failed to download tokenizer.json: {e:?}"),
        }
        match repo.get("tokenizer_config.json") {
            Ok(p) => info!("Downloaded tokenizer_config.json to {}", p.display()),
            Err(e) => warn!("Failed to download tokenizer_config.json: {e:?}"),
        }
        // Download quantization config files if config.json indicates GPTQ or AWQ.
        if let Ok(config_str) = std::fs::read_to_string(&config_path) {
            if config_str.contains("\"gptq\"")
                && let Ok(p) = repo.get("quantize_config.json")
            {
                info!("Downloaded quantize_config.json to {}", p.display());
            }
            if config_str.contains("\"awq\"")
                && let Ok(p) = repo.get("quant_config.json")
            {
                info!("Downloaded quant_config.json to {}", p.display());
            }
        }

        // Download weights.
        if repo.get("model.safetensors").is_ok() {
            info!("Downloaded single safetensors file");
            return Ok(model_dir);
        }

        // Sharded weights.
        if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            info!("Downloading sharded model weights...");
            let index = vllm_model::weight::SafeTensorsIndex::from_file(&index_path)
                .map_err(|e| ExecutorError::WorkerInit(format!("failed to parse index: {e}")))?;

            let sorted_shards = index.shard_files();
            let total = sorted_shards.len();

            // Filter out shards already present in the cache.
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
                                    repo.download_with_progress(shard, bar).map(|_| ()).map_err(
                                        |e| {
                                            ExecutorError::WorkerInit(format!(
                                                "failed to download {shard}: {e}"
                                            ))
                                        },
                                    )
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

        Ok(model_dir)
    }
}

impl Worker for MlxWorker {
    fn init_device(&mut self) -> ExecutorResult<()> {
        // MLX defaults to GPU on Apple Silicon. No explicit device init needed.
        info!("MlxWorker: initialized (MLX Metal GPU backend)");
        Ok(())
    }

    fn load_model(&mut self) -> ExecutorResult<()> {
        let explicit_dtype = self.config.mlx_dtype()?;

        // Resolve model directory.
        let model_dir = self.resolve_model_path()?;
        info!("MlxWorker: loading model from {}", model_dir.display());

        // Optimistically parse tokenizer.json on a background thread while we
        // load config + weights on the main thread.
        let tok_dir = model_dir.clone();
        let tokenizer_handle = std::thread::spawn(move || {
            let path = tok_dir.join("tokenizer.json");
            if path.exists() {
                tokenizers::Tokenizer::from_file(&path).ok()
            } else {
                None
            }
        });

        // Parse config.json.
        let hf_config = HfModelConfig::from_dir(&model_dir)
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to parse config.json: {e}")))?;

        // Unwrap composite models (e.g. Kimi K2.5 → text_config).
        let hf_config = match hf_config.resolve_text_config() {
            Some((text_cfg, prefix)) => {
                info!(
                    "MlxWorker: composite model detected, unwrapping text config (strip prefix: {prefix})"
                );
                text_cfg
            }
            None => hf_config,
        };

        // Resolve dtype.
        let dtype = if let Some(dt) = explicit_dtype {
            dt
        } else {
            let resolved = hf_config
                .torch_dtype
                .as_deref()
                .and_then(|s| match s {
                    "float16" => Some(Dtype::Float16),
                    "bfloat16" => Some(Dtype::Bfloat16),
                    "float32" => Some(Dtype::Float32),
                    _ => None,
                })
                .unwrap_or(Dtype::Float16);
            info!("MlxWorker: auto dtype resolved to {:?}", resolved);
            resolved
        };

        // Detect quantization from config.json.
        let is_quantized = hf_config.extra.contains_key("quantization");
        if is_quantized {
            let qinfo = hf_config.extra.get("quantization").unwrap();
            info!("MlxWorker: detected quantized model: {qinfo}");
        }

        // Detect GPTQ/AWQ quantization.
        let quant_method = hf_config
            .extra
            .get("quantization_config")
            .and_then(|v| v.get("quant_method"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let is_gptq = vllm_model::gptq_config::GptqQuantizeConfig::from_dir(&model_dir).is_ok()
            || quant_method.as_deref() == Some("gptq");

        let is_awq = !is_gptq
            && (vllm_model::awq_config::AwqQuantizeConfig::from_dir(&model_dir).is_ok()
                || quant_method.as_deref() == Some("awq"));

        let is_bnb = !is_gptq && !is_awq && quant_method.as_deref() == Some("bitsandbytes");

        // Look up architecture in the MLX registry.
        let arch = hf_config
            .architectures
            .first()
            .ok_or_else(|| {
                ExecutorError::WorkerInit("config.json has no architectures field".to_string())
            })?
            .clone();
        let registry = MlxModelRegistry::default_registry();

        let factory = if is_gptq {
            info!("MlxWorker: GPTQ quantization detected, using dequantize-at-load path");
            registry.get_gptq(&arch).ok_or_else(|| {
                ExecutorError::WorkerInit(format!("unsupported GPTQ MLX architecture: {arch}"))
            })?
        } else if is_awq {
            info!("MlxWorker: AWQ quantization detected, using dequantize-at-load path");
            registry.get_awq(&arch).ok_or_else(|| {
                ExecutorError::WorkerInit(format!("unsupported AWQ MLX architecture: {arch}"))
            })?
        } else if is_bnb {
            info!("MlxWorker: BnB quantization detected, using dequantize-at-load path");
            registry.get_bnb(&arch).ok_or_else(|| {
                ExecutorError::WorkerInit(format!("unsupported BnB MLX architecture: {arch}"))
            })?
        } else {
            registry.get_factory(&arch, is_quantized).ok_or_else(|| {
                ExecutorError::WorkerInit(format!(
                    "unsupported MLX architecture: {arch}. Supported: {:?}",
                    registry.architectures().collect::<Vec<_>>()
                ))
            })?
        };

        // Load the model.
        let model = factory(&model_dir, &hf_config, dtype).map_err(|e| {
            ExecutorError::WorkerInit(format!("failed to construct MLX model: {e}"))
        })?;

        self.model_dir = Some(model_dir);
        self.hf_config = Some(hf_config);
        self.resolved_dtype = Some(dtype);
        self.model = Some(model);
        self.resolved_architecture = Some(arch.clone());
        let quant_str = if is_gptq {
            ", GPTQ"
        } else if is_awq {
            ", AWQ"
        } else if is_bnb {
            ", BnB"
        } else if is_quantized {
            ", quantized"
        } else {
            ""
        };
        info!("MlxWorker: model loaded (arch={arch}, dtype={dtype:?}{quant_str})");

        // Inject LoRA adapter if configured.
        if let Some(ref adapter_path) = self.config.lora_adapter {
            let adapter_dir = std::path::Path::new(adapter_path);
            if !adapter_dir.exists() {
                return Err(ExecutorError::WorkerInit(format!(
                    "LoRA adapter path does not exist: {}",
                    adapter_path
                )));
            }
            let adapter = crate::lora::MlxLoraAdapter::from_dir(adapter_dir, "default", dtype)
                .map_err(|e| {
                    ExecutorError::WorkerInit(format!("failed to load LoRA adapter: {e}"))
                })?;
            if let Some(ref mut model) = self.model {
                model.inject_lora(&adapter).map_err(|e| {
                    ExecutorError::WorkerInit(format!("failed to inject LoRA: {e}"))
                })?;
            }
            info!(
                "MlxWorker: LoRA adapter '{}' loaded (rank={}, targets={:?})",
                adapter.name, adapter.config.r, adapter.config.target_modules
            );
        }
        // Resolve pooling strategy for embeddings.
        self.resolve_pooling_strategy();

        // Grammar vocabulary is built lazily on first constrained-decoding request.

        // Collect the tokenizer we loaded in the background.
        self.preloaded_tokenizer = tokenizer_handle.join().unwrap_or(None);

        Ok(())
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        _num_cpu_blocks: usize,
    ) -> ExecutorResult<()> {
        self.num_gpu_blocks = num_gpu_blocks;
        info!(
            "MlxWorker: cache initialized (gpu_blocks={})",
            num_gpu_blocks
        );
        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        // Apple Silicon uses unified memory — GPU and CPU share the same pool.
        // Report total physical memory (not just "available") since MLX manages
        // its own memory pool and the OS will page out inactive data as needed.
        //
        // Use sysctl directly instead of the `sysinfo` crate, which is
        // surprisingly expensive (~10-50ms) due to process enumeration.
        let total = unsafe {
            let mut memsize: u64 = 0;
            let mut size = std::mem::size_of::<u64>();
            let mut mib = [libc::CTL_HW, libc::HW_MEMSIZE];
            libc::sysctl(
                mib.as_mut_ptr(),
                2,
                &mut memsize as *mut u64 as *mut libc::c_void,
                &mut size,
                std::ptr::null_mut(),
                0,
            );
            memsize as usize
        };
        if total == 0 {
            return Ok(8 * 1024 * 1024 * 1024);
        }
        Ok(total)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        // --- Double-buffer flush: read PREVIOUS step's GPU results ---
        // These were async_eval'd last call.  By now the GPU is done (the main
        // thread did finalize + schedule in between).  as_slice is instant.
        let prev_output = if let Some(pending) = self.pending_step.take() {
            let mut token_map: HashMap<String, Vec<u32>> = HashMap::new();
            if let Some(ref greedy_result) = pending.greedy_result {
                let flat = greedy_result.as_slice::<u32>();
                for (req_id, batch_pos) in &pending.greedy_mapping {
                    let token_id = flat[*batch_pos];
                    token_map.insert(req_id.clone(), vec![token_id]);
                    if let Some(buf) = self.token_buffers.get_mut(req_id) {
                        buf.push(token_id);
                    }
                    #[cfg(feature = "guided-decoding")]
                    if let Some(guide) = self.grammar_states.get_mut(req_id) {
                        guide.advance(token_id);
                    }
                }
            }
            for (mapping, arr) in &pending.temp_results {
                let flat = arr.as_slice::<u32>();
                for (req_id, batch_pos) in mapping {
                    let token_id = flat[*batch_pos];
                    token_map.insert(req_id.clone(), vec![token_id]);
                    if let Some(buf) = self.token_buffers.get_mut(req_id) {
                        buf.push(token_id);
                    }
                    #[cfg(feature = "guided-decoding")]
                    if let Some(guide) = self.grammar_states.get_mut(req_id) {
                        guide.advance(token_id);
                    }
                }
            }
            Some(ModelRunnerOutput::from_token_map(token_map))
        } else {
            None
        };

        // Lazily build grammar vocabulary if any new request needs constrained decoding.
        // Done before borrowing self.model to satisfy the borrow checker.
        #[cfg(feature = "guided-decoding")]
        {
            let needs_grammar = scheduler_output.scheduled_new_reqs.iter().any(|r| {
                r.sampling_params
                    .as_ref()
                    .is_some_and(|p| p.guided_grammar.is_some())
            });
            if needs_grammar {
                self.ensure_grammar_factory();
            }
        }

        let model = self
            .model
            .as_mut()
            .ok_or_else(|| ExecutorError::WorkerExecution("model not loaded".to_string()))?;

        let num_layers = model.num_layers();

        // Clean up finished requests — stash KV caches for prefix reuse.
        // If a finished request's cache lives in the persistent batched cache,
        // extract it first, then invalidate the batched cache.
        if !scheduler_output.finished_req_ids.is_empty() {
            let any_finished_in_batch = self.batched_decode_cache.as_ref().is_some_and(|bdc| {
                scheduler_output
                    .finished_req_ids
                    .iter()
                    .any(|rid| bdc.req_ids.contains(rid))
            });

            if any_finished_in_batch {
                // Extract ALL individual caches from the batch.
                // Finished ones will be cleaned up below; remaining ones
                // are available for the next batch formation.
                let bdc = self.batched_decode_cache.take().unwrap();
                for (idx, rid) in bdc.req_ids.iter().enumerate() {
                    let mut per_req_kv: MlxKvCache = Vec::with_capacity(num_layers);
                    for layer_idx in 0..num_layers {
                        let left_pad = bdc.left_padding[layer_idx][idx];
                        let layer_cache = bdc.layer_caches[layer_idx]
                            .extract_individual(idx, left_pad)
                            .ok();
                        per_req_kv.push(layer_cache);
                    }
                    self.kv_caches.insert(rid.clone(), per_req_kv);
                }
            }
        }

        for req_id in &scheduler_output.finished_req_ids {
            if self.enable_prefix_caching {
                if let (Some(prompt), Some(kv_cache)) = (
                    self.token_buffers.get(req_id),
                    self.kv_caches.remove(req_id),
                ) {
                    if prompt.len() >= self.prefix_block_size {
                        let h = hash_prefix(prompt, self.prefix_block_size);
                        if !self.kv_cache_pool.contains_key(&h) {
                            // FIFO eviction when pool is full.
                            if self.kv_cache_pool.len() >= PREFIX_CACHE_POOL_MAX
                                && let Some(&oldest) = self.kv_cache_pool.keys().next()
                            {
                                self.kv_cache_pool.remove(&oldest);
                            }
                            self.kv_cache_pool.insert(h, kv_cache);
                        }
                    }
                } else {
                    self.kv_caches.remove(req_id);
                }
            } else {
                self.kv_caches.remove(req_id);
            }
            self.token_buffers.remove(req_id);
            self.sampling_params_map.remove(req_id);
            self.recurrent_states.remove(req_id);
            #[cfg(feature = "guided-decoding")]
            self.grammar_states.remove(req_id);
            self.mm_data_map.remove(req_id);
        }

        // Collect requests.
        struct ReqInput {
            req_id: String,
            token_ids: Vec<u32>,
            positions: Vec<i32>,
            is_prefill: bool,
            /// Draft tokens for speculative decode verification.
            spec_token_ids: Vec<u32>,
        }
        let mut req_inputs: Vec<ReqInput> = Vec::new();

        // New requests (prefill).
        for new_req in &scheduler_output.scheduled_new_reqs {
            let num_tokens = scheduler_output
                .num_scheduled_tokens
                .get(&new_req.req_id)
                .copied()
                .unwrap_or(0);
            if num_tokens == 0 {
                continue;
            }

            let prompt_ids = new_req.prompt_token_ids.as_deref().unwrap_or(&[]);
            let num_computed = new_req.num_computed_tokens as usize;

            // Fix: slice from num_computed_tokens, not 0.
            // The scheduler tells us how many prefix tokens are already cached;
            // we only need to forward the remaining tokens.
            let start = num_computed.min(prompt_ids.len());
            let end = (start + num_tokens).min(prompt_ids.len());
            let tokens_to_use = &prompt_ids[start..end];

            self.token_buffers
                .insert(new_req.req_id.clone(), prompt_ids.to_vec());
            if let Some(ref params) = new_req.sampling_params {
                // Create grammar guide for constrained decoding if requested.
                #[cfg(feature = "guided-decoding")]
                if let Some(ref grammar) = params.guided_grammar {
                    if let Some(ref factory) = self.grammar_factory {
                        match vllm_model::grammar::GrammarGuide::from_guided_grammar(
                            grammar, factory,
                        ) {
                            Ok(guide) => {
                                self.grammar_states.insert(new_req.req_id.clone(), guide);
                            }
                            Err(e) => {
                                warn!(
                                    "Failed to compile grammar for request {}: {e}",
                                    new_req.req_id
                                );
                            }
                        }
                    } else {
                        warn!(
                            "Grammar requested for {} but no vocabulary available",
                            new_req.req_id
                        );
                    }
                }
                self.sampling_params_map
                    .insert(new_req.req_id.clone(), params.clone());
            }

            // Try to reuse a cached KV from the prefix pool.
            let kv_cache = if num_computed > 0 && self.enable_prefix_caching {
                let prefix = &prompt_ids[..num_computed];
                let h = hash_prefix(prefix, self.prefix_block_size);
                if let Some(cached) = self.kv_cache_pool.get(&h) {
                    // Clone (MLX copy-on-write) and truncate to the matched prefix length.
                    let mut kv = cached.clone();
                    for layer in kv.iter_mut().flatten() {
                        if layer.seq_len() > num_computed {
                            layer.truncate(num_computed);
                        }
                    }
                    debug!(
                        "prefix cache hit for req {} ({} computed tokens)",
                        new_req.req_id, num_computed
                    );
                    kv
                } else {
                    debug!(
                        "prefix cache miss for req {} ({} computed tokens, forwarding all)",
                        new_req.req_id, num_computed
                    );
                    cache::empty_kv_cache(num_layers)
                }
            } else {
                cache::empty_kv_cache(num_layers)
            };
            self.kv_caches.insert(new_req.req_id.clone(), kv_cache);

            let pos_offset = num_computed as i32;
            let positions: Vec<i32> = (0..tokens_to_use.len() as i32)
                .map(|i| pos_offset + i)
                .collect();

            // Store multimodal data for VLM models (consumed during forward).
            if let Some(mm_data) = new_req.mm_data.clone() {
                self.mm_data_map.insert(new_req.req_id.clone(), mm_data);
            }

            req_inputs.push(ReqInput {
                req_id: new_req.req_id.clone(),
                token_ids: tokens_to_use.to_vec(),
                positions,
                is_prefill: true,
                spec_token_ids: Vec::new(),
            });
        }

        // Cached requests (decode).
        let new_req_ids: HashSet<&str> = scheduler_output
            .scheduled_new_reqs
            .iter()
            .map(|r| r.req_id.as_str())
            .collect();

        for req_id in scheduler_output.scheduled_cached_reqs.req_ids.iter() {
            if new_req_ids.contains(req_id.as_str()) {
                continue;
            }
            let num_tokens = scheduler_output
                .num_scheduled_tokens
                .get(req_id)
                .copied()
                .unwrap_or(0);
            if num_tokens == 0 {
                continue;
            }

            if let Some(buf) = self.token_buffers.get(req_id) {
                let last_token = *buf.last().unwrap_or(&0);
                let position = (buf.len() - 1) as i32;

                // Check for speculative decode draft tokens.
                let spec_tokens = scheduler_output
                    .scheduled_spec_decode_tokens
                    .get(req_id)
                    .cloned()
                    .unwrap_or_default();

                if spec_tokens.is_empty() {
                    req_inputs.push(ReqInput {
                        req_id: req_id.clone(),
                        token_ids: vec![last_token],
                        positions: vec![position],
                        is_prefill: false,
                        spec_token_ids: Vec::new(),
                    });
                } else {
                    // Speculative decode: feed [last_token, draft_1, ..., draft_K].
                    let mut token_ids = Vec::with_capacity(1 + spec_tokens.len());
                    let mut positions = Vec::with_capacity(1 + spec_tokens.len());
                    token_ids.push(last_token);
                    positions.push(position);
                    for (j, &draft_tok) in spec_tokens.iter().enumerate() {
                        token_ids.push(draft_tok);
                        positions.push(position + 1 + j as i32);
                    }
                    req_inputs.push(ReqInput {
                        req_id: req_id.clone(),
                        token_ids,
                        positions,
                        is_prefill: false,
                        spec_token_ids: spec_tokens,
                    });
                }
            } else {
                warn!("No token buffer for continuing request {}", req_id);
            }
        }

        // Backward compat: handle requests only in num_scheduled_tokens.
        for req_id in scheduler_output.num_scheduled_tokens.keys() {
            if new_req_ids.contains(req_id.as_str()) {
                continue;
            }
            if req_inputs.iter().any(|r| r.req_id == *req_id) {
                continue;
            }
            let num_tokens = scheduler_output.num_scheduled_tokens[req_id];
            if num_tokens == 0 {
                continue;
            }
            if let Some(buf) = self.token_buffers.get(req_id) {
                let last_token = *buf.last().unwrap_or(&0);
                let position = (buf.len() - 1) as i32;
                req_inputs.push(ReqInput {
                    req_id: req_id.clone(),
                    token_ids: vec![last_token],
                    positions: vec![position],
                    is_prefill: false,
                    spec_token_ids: Vec::new(),
                });
            }
        }

        if req_inputs.is_empty() {
            return Ok(ModelRunnerOutput::from_token_map(HashMap::new()));
        }

        // --- Pooling mode: run hidden_states + pool + normalize ---
        if self.is_pooling {
            use vllm_model::embedding::PoolingStrategy;
            let strategy = self.pooling_strategy;

            let mut pooler_map: HashMap<String, Vec<f32>> = HashMap::new();

            for ri in &req_inputs {
                let token_ids = match self.token_buffers.get(&ri.req_id) {
                    Some(buf) => buf.as_slice(),
                    None => continue,
                };

                let input_ids = Array::from_iter(
                    token_ids.iter().map(|&t| t as i32),
                    &[token_ids.len() as i32],
                );
                let positions = Array::from_iter(
                    (0..token_ids.len() as i32).collect::<Vec<_>>(),
                    &[token_ids.len() as i32],
                );

                let hidden = model
                    .hidden_states(&input_ids, &positions)
                    .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

                // Pool according to strategy.
                let num_tokens = token_ids.len() as i32;
                let pooled = match strategy {
                    PoolingStrategy::Last => hidden.index(num_tokens - 1),
                    PoolingStrategy::Cls => hidden.index(0),
                    PoolingStrategy::Mean => {
                        let sum = hidden
                            .sum_axis(0, None)
                            .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
                        sum.divide(Array::from(num_tokens as f32))
                            .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?
                    }
                };

                // L2 normalize.
                let sq = pooled
                    .square()
                    .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
                let sum = sq
                    .sum(None)
                    .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
                let norm = sum
                    .sqrt()
                    .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
                let normalized = pooled
                    .divide(&norm)
                    .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
                let normalized_f32 = normalized
                    .as_dtype(Dtype::Float32)
                    .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

                normalized_f32
                    .eval()
                    .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
                let vec: Vec<f32> = normalized_f32.as_slice().to_vec();

                pooler_map.insert(ri.req_id.clone(), vec);
            }

            let req_ids: Vec<String> = pooler_map.keys().cloned().collect();
            let req_id_to_index: HashMap<String, usize> = req_ids
                .iter()
                .enumerate()
                .map(|(i, id)| (id.clone(), i))
                .collect();
            let sampled_token_ids = vec![vec![]; req_ids.len()];

            return Ok(ModelRunnerOutput {
                req_ids,
                req_id_to_index,
                sampled_token_ids,
                logprobs: None,
                prompt_logprobs_dict: HashMap::new(),
                draft_token_ids: None,
                pooler_output: Some(pooler_map),
                d2h_resolver: None,
            });
        }

        let mut token_map: HashMap<String, Vec<u32>> = HashMap::new();
        let mut logprobs_map: HashMap<String, Vec<vllm_common::LogprobsOutput>> = HashMap::new();
        let mut prompt_logprobs_map: HashMap<String, Vec<vllm_common::LogprobsOutput>> =
            HashMap::new();
        let any_logprobs_requested = req_inputs.iter().any(|r| {
            self.sampling_params_map
                .get(&r.req_id)
                .and_then(|p| p.logprobs)
                .is_some()
        });

        // CPU sampler for penalty/min_p/logit_bias/logprobs support.
        let mut cpu_sampler = vllm_model::Sampler::new();

        let step_start = Instant::now();

        // --- Phase A: Run forward passes, collecting lazy logit arrays ---
        // MLX lazy eval means these build compute graphs without materializing.
        // By deferring eval until after all forwards, MLX can fuse all graphs
        // into a single Metal command buffer.
        //
        // Batched path: requests without recurrent state or mm_data are grouped
        // into a single `forward_batch` call. Remaining requests use per-request `forward`.
        struct LazyReqOutput {
            last_logits: Array,
            full_logits_f32: Option<Array>,
            prompt_logprobs_info: Option<(usize, Vec<u32>)>,
        }
        let mut lazy_outputs: Vec<LazyReqOutput> = Vec::with_capacity(req_inputs.len());

        // Partition requests into batchable vs fallback.
        let has_recurrent = model.num_recurrent_layers() > 0;
        let mut batch_indices: Vec<usize> = Vec::new();
        let mut fallback_indices: Vec<usize> = Vec::new();
        for (idx, ri) in req_inputs.iter().enumerate() {
            let needs_fallback = has_recurrent || self.mm_data_map.contains_key(&ri.req_id);
            if needs_fallback {
                fallback_indices.push(idx);
            } else {
                batch_indices.push(idx);
            }
        }

        // --- Fast path: single greedy decode request (the ITL-critical path) ---
        // When there's exactly one batchable request doing a single-token greedy decode
        // with no special needs, use forward_greedy() to fuse forward + argmax into one
        // lazy graph. MLX can then avoid materializing the full vocab-size logits tensor.
        if batch_indices.len() == 1 && fallback_indices.is_empty() {
            let ri = &req_inputs[batch_indices[0]];
            let is_single_token = ri.token_ids.len() == 1;
            let no_spec = ri.spec_token_ids.is_empty();
            let no_prompt_logprobs = !ri.is_prefill
                || self
                    .sampling_params_map
                    .get(&ri.req_id)
                    .and_then(|p| p.prompt_logprobs)
                    .is_none();
            let params = self.sampling_params_map.get(&ri.req_id);
            let is_greedy = params
                .map(|p| (p.temperature as f32) < 1e-5)
                .unwrap_or(true);

            #[cfg(feature = "guided-decoding")]
            let has_grammar = self
                .grammar_states
                .get_mut(&ri.req_id)
                .is_some_and(|g| g.allowed_tokens().is_some());
            #[cfg(not(feature = "guided-decoding"))]
            let has_grammar = false;

            let no_cpu_needs = !has_grammar
                && params
                    .map(|p| {
                        p.repetition_penalty == 1.0
                            && p.frequency_penalty == 0.0
                            && p.presence_penalty == 0.0
                            && p.min_p <= 0.0
                            && p.logit_bias.is_none()
                            && p.logprobs.is_none()
                            && p.top_k <= 0
                            && p.top_p >= 1.0
                    })
                    .unwrap_or(true);

            if is_single_token && no_spec && no_prompt_logprobs && is_greedy && no_cpu_needs {
                let req_id = &ri.req_id;

                let input_ids = Array::from_iter(
                    ri.token_ids.iter().map(|&t| t as i32),
                    &[ri.token_ids.len() as i32],
                );
                let positions =
                    Array::from_iter(ri.positions.iter().copied(), &[ri.positions.len() as i32]);
                let rope_offset = ri.positions.iter().copied().min();

                let kv_cache = self
                    .kv_caches
                    .entry(req_id.clone())
                    .or_insert_with(|| cache::empty_kv_cache(num_layers));

                // forward_greedy: forward + argmax fused into single lazy graph.
                let token_ids_arr = model
                    .forward_greedy(&input_ids, &positions, kv_cache, rope_offset)
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("forward_greedy failed: {e}"))
                    })?;

                // Single eval: entire forward + argmax in one Metal command buffer.
                token_ids_arr
                    .eval()
                    .map_err(|e| ExecutorError::WorkerExecution(format!("eval error: {e}")))?;

                let flat = token_ids_arr.as_slice::<u32>();
                token_map.insert(req_id.clone(), flat.to_vec());

                // Update token buffer.
                if let Some(buf) = self.token_buffers.get_mut(req_id) {
                    buf.extend_from_slice(flat);
                }

                // Advance grammar state (none in this fast path, but be consistent).
                #[cfg(feature = "guided-decoding")]
                if let Some(guide) = self.grammar_states.get_mut(req_id)
                    && let Some(&token_id) = flat.first()
                {
                    guide.advance(token_id);
                }

                let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                self.step_count += 1;
                if ri.is_prefill {
                    self.prefill_count += 1;
                    self.total_prefill_ms += step_ms;
                } else {
                    self.decode_count += 1;
                    self.total_decode_ms += step_ms;
                }

                let output = ModelRunnerOutput::from_token_map(token_map);
                return Ok(output);
            }
        }

        // Pre-allocate output slots.
        for _ in 0..req_inputs.len() {
            lazy_outputs.push(LazyReqOutput {
                last_logits: Array::from_f32(0.0),
                full_logits_f32: None,
                prompt_logprobs_info: None,
            });
        }

        // --- Batched forward for eligible requests ---
        if !batch_indices.is_empty() {
            // Build flat input_ids, positions, and batch_info.
            let mut flat_token_ids: Vec<i32> = Vec::new();
            let mut flat_positions: Vec<i32> = Vec::new();
            let mut q_lens: Vec<usize> = Vec::new();
            let mut rope_offsets: Vec<i32> = Vec::new();

            for &idx in &batch_indices {
                let ri = &req_inputs[idx];
                for &t in &ri.token_ids {
                    flat_token_ids.push(t as i32);
                }
                flat_positions.extend_from_slice(&ri.positions);
                q_lens.push(ri.token_ids.len());
                rope_offsets.push(ri.positions.iter().copied().min().unwrap_or(0));
            }

            let batch_info = cache::MlxBatchInfo::new(q_lens, rope_offsets);
            let total = batch_info.total_tokens as i32;
            let input_ids_arr = Array::from_iter(flat_token_ids, &[total]);
            let positions_arr = Array::from_iter(flat_positions, &[total]);

            // Check if we can use the persistent batched decode cache.
            let batch_req_ids: Vec<String> = batch_indices
                .iter()
                .map(|&idx| req_inputs[idx].req_id.clone())
                .collect();
            let all_decode = batch_info.q_lens.iter().all(|&ql| ql == 1)
                && batch_indices.len() > 1
                && model.supports_batch_decode();
            let cache_matches = self
                .batched_decode_cache
                .as_ref()
                .is_some_and(|c| c.req_ids == batch_req_ids);

            let all_logits = if all_decode && cache_matches {
                // === Fast path: reuse persistent batched cache ===
                // Only a single slice_update per layer (no from_individual, no write_back).
                // Take the cache out to satisfy the borrow checker (model borrows self.model,
                // bdc borrows self.batched_decode_cache — must not overlap through self).
                let mut bdc = self.batched_decode_cache.take().unwrap();
                let result = model.forward_batch_decode(
                    &input_ids_arr,
                    &batch_info,
                    &mut bdc.layer_caches,
                    &bdc.left_padding,
                );
                // Put the cache back BEFORE propagating errors.
                self.batched_decode_cache = Some(bdc);
                result.map_err(|e| {
                    ExecutorError::WorkerExecution(format!("batched decode forward failed: {e}"))
                })?
            } else if all_decode {
                // === Build new persistent batched cache ===
                // First, invalidate existing batched cache, writing back to per-request.
                if let Some(old_bdc) = self.batched_decode_cache.take() {
                    for (idx, rid) in old_bdc.req_ids.iter().enumerate() {
                        let mut per_req_kv: MlxKvCache = Vec::with_capacity(num_layers);
                        for layer_idx in 0..num_layers {
                            let left_pad = old_bdc.left_padding[layer_idx][idx];
                            let layer_cache = old_bdc.layer_caches[layer_idx]
                                .extract_individual(idx, left_pad)
                                .ok();
                            per_req_kv.push(layer_cache);
                        }
                        self.kv_caches.insert(rid.clone(), per_req_kv);
                    }
                }

                // Re-check after invalidation.
                let all_have_caches = batch_req_ids.iter().all(|rid| {
                    self.kv_caches
                        .get(rid)
                        .is_some_and(|kv| !kv.is_empty() && kv.iter().all(|c| c.is_some()))
                });

                if all_have_caches {
                    // Build batched caches for ALL layers.
                    let mut layer_caches: Vec<BatchMlxLayerKvCache> =
                        Vec::with_capacity(num_layers);
                    let mut left_paddings: Vec<Vec<usize>> = Vec::with_capacity(num_layers);

                    for layer_idx in 0..num_layers {
                        let cache_refs: Vec<&cache::MlxLayerKvCache> = batch_req_ids
                            .iter()
                            .map(|rid| {
                                self.kv_caches.get(rid).unwrap()[layer_idx]
                                    .as_ref()
                                    .expect("decode requires cache")
                            })
                            .collect();
                        let max_seq = cache_refs.iter().map(|c| c.seq_len()).max().unwrap_or(0);
                        let left_pad: Vec<usize> =
                            cache_refs.iter().map(|c| max_seq - c.seq_len()).collect();
                        layer_caches.push(
                            BatchMlxLayerKvCache::from_individual(&cache_refs).map_err(|e| {
                                ExecutorError::WorkerExecution(format!(
                                    "batch cache build failed: {e}"
                                ))
                            })?,
                        );
                        left_paddings.push(left_pad);
                    }

                    let mut bdc = BatchedDecodeCache {
                        req_ids: batch_req_ids.clone(),
                        layer_caches,
                        left_padding: left_paddings,
                    };

                    // Run model with the new batched cache.
                    let logits = model
                        .forward_batch_decode(
                            &input_ids_arr,
                            &batch_info,
                            &mut bdc.layer_caches,
                            &bdc.left_padding,
                        )
                        .map_err(|e| {
                            ExecutorError::WorkerExecution(format!(
                                "batched decode forward failed: {e}"
                            ))
                        })?;

                    // Store the persistent batched cache.
                    self.batched_decode_cache = Some(bdc);

                    // Remove per-request caches (they're now in the batch).
                    for rid in &batch_req_ids {
                        if let Some(slot) = self.kv_caches.get_mut(rid) {
                            *slot = Vec::new();
                        }
                    }

                    logits
                } else {
                    // Some per-request caches missing — fall back to per-request path.
                    let mut batch_kv_caches: Vec<cache::MlxKvCache> = batch_indices
                        .iter()
                        .map(|&idx| {
                            let req_id = &req_inputs[idx].req_id;
                            self.kv_caches
                                .get_mut(req_id)
                                .map(std::mem::take)
                                .unwrap_or_else(|| cache::empty_kv_cache(num_layers))
                        })
                        .collect();
                    let logits = model
                        .forward_batch(
                            &input_ids_arr,
                            &positions_arr,
                            &batch_info,
                            &mut batch_kv_caches,
                        )
                        .map_err(|e| {
                            ExecutorError::WorkerExecution(format!("batched forward failed: {e}"))
                        })?;
                    for &idx in &batch_indices {
                        let kv = batch_kv_caches.remove(0);
                        if let Some(slot) = self.kv_caches.get_mut(&req_inputs[idx].req_id) {
                            *slot = kv;
                        } else {
                            self.kv_caches.insert(req_inputs[idx].req_id.clone(), kv);
                        }
                    }
                    logits
                }
            } else {
                // === Per-request path (prefill, mixed, or single request) ===
                // Invalidate persistent batched cache if it exists.
                if let Some(old_bdc) = self.batched_decode_cache.take() {
                    for (idx, rid) in old_bdc.req_ids.iter().enumerate() {
                        let mut per_req_kv: MlxKvCache = Vec::with_capacity(num_layers);
                        for layer_idx in 0..num_layers {
                            let left_pad = old_bdc.left_padding[layer_idx][idx];
                            let layer_cache = old_bdc.layer_caches[layer_idx]
                                .extract_individual(idx, left_pad)
                                .ok();
                            per_req_kv.push(layer_cache);
                        }
                        self.kv_caches.insert(rid.clone(), per_req_kv);
                    }
                }

                let mut batch_kv_caches: Vec<cache::MlxKvCache> = batch_indices
                    .iter()
                    .map(|&idx| {
                        let req_id = &req_inputs[idx].req_id;
                        self.kv_caches
                            .get_mut(req_id)
                            .map(std::mem::take)
                            .unwrap_or_else(|| cache::empty_kv_cache(num_layers))
                    })
                    .collect();
                let logits = model
                    .forward_batch(
                        &input_ids_arr,
                        &positions_arr,
                        &batch_info,
                        &mut batch_kv_caches,
                    )
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("batched forward failed: {e}"))
                    })?;
                for &idx in &batch_indices {
                    let kv = batch_kv_caches.remove(0);
                    if let Some(slot) = self.kv_caches.get_mut(&req_inputs[idx].req_id) {
                        *slot = kv;
                    } else {
                        self.kv_caches.insert(req_inputs[idx].req_id.clone(), kv);
                    }
                }
                logits
            };

            // Split output logits per-request.
            for (i, &idx) in batch_indices.iter().enumerate() {
                let ri = &req_inputs[idx];
                let start = batch_info.offsets[i] as i32;
                let len = batch_info.q_lens[i] as i32;

                // Extract per-request logits from the flat output.
                let req_logits = if batch_indices.len() == 1 {
                    all_logits.clone()
                } else {
                    all_logits.index((start..start + len, ..))
                };

                let last_pos = ri.token_ids.len() - 1;
                let last_logits = if ri.token_ids.len() > 1 {
                    req_logits
                        .index(last_pos as i32)
                        .expand_dims(0)
                        .and_then(|a| a.as_dtype(Dtype::Float32))
                        .map_err(|e| ExecutorError::WorkerExecution(format!("index error: {e}")))?
                } else {
                    req_logits
                        .as_dtype(Dtype::Float32)
                        .map_err(|e| ExecutorError::WorkerExecution(format!("dtype cast: {e}")))?
                };

                let needs_full_logits = (ri.is_prefill
                    && ri.token_ids.len() > 1
                    && self
                        .sampling_params_map
                        .get(&ri.req_id)
                        .and_then(|p| p.prompt_logprobs)
                        .is_some())
                    || !ri.spec_token_ids.is_empty();

                let (full_logits_f32, prompt_logprobs_info) = if needs_full_logits {
                    let top_n = self
                        .sampling_params_map
                        .get(&ri.req_id)
                        .and_then(|p| p.prompt_logprobs)
                        .map(|n| n.max(0) as usize);
                    let f32_logits = req_logits.as_dtype(Dtype::Float32).map_err(|e| {
                        ExecutorError::WorkerExecution(format!("dtype cast error: {e}"))
                    })?;
                    let info = top_n.map(|n| (n, ri.token_ids.clone()));
                    (Some(f32_logits), info)
                } else {
                    (None, None)
                };

                lazy_outputs[idx] = LazyReqOutput {
                    last_logits,
                    full_logits_f32,
                    prompt_logprobs_info,
                };
            }
        }

        // --- Fallback: per-request forward for requests with recurrent state or mm_data ---
        for &idx in &fallback_indices {
            let req_input = &req_inputs[idx];

            if model.num_recurrent_layers() > 0 {
                if let Some(rs) = self.recurrent_states.get(&req_input.req_id) {
                    model.inject_recurrent_state(rs);
                } else {
                    model.reset_recurrent_state();
                }
            }

            let input_ids = Array::from_iter(
                req_input.token_ids.iter().map(|&t| t as i32),
                &[req_input.token_ids.len() as i32],
            );
            let positions = Array::from_iter(
                req_input.positions.iter().copied(),
                &[req_input.positions.len() as i32],
            );

            let kv_cache = self
                .kv_caches
                .entry(req_input.req_id.clone())
                .or_insert_with(|| cache::empty_kv_cache(num_layers));

            let mm_data = self.mm_data_map.remove(&req_input.req_id);
            model.set_mm_data(mm_data);

            let rope_offset = req_input.positions.iter().copied().min();

            let logits = model
                .forward(&input_ids, &positions, kv_cache, rope_offset)
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("model forward failed: {e}"))
                })?;

            if model.num_recurrent_layers() > 0 {
                self.recurrent_states
                    .insert(req_input.req_id.clone(), model.extract_recurrent_state());
            }

            let last_pos = req_input.token_ids.len() - 1;
            let last_logits = if req_input.token_ids.len() > 1 {
                logits
                    .index(last_pos as i32)
                    .expand_dims(0)
                    .and_then(|a| a.as_dtype(Dtype::Float32))
                    .map_err(|e| ExecutorError::WorkerExecution(format!("index error: {e}")))?
            } else {
                logits
                    .as_dtype(Dtype::Float32)
                    .map_err(|e| ExecutorError::WorkerExecution(format!("dtype cast: {e}")))?
            };

            let needs_full_logits = (req_input.is_prefill
                && req_input.token_ids.len() > 1
                && self
                    .sampling_params_map
                    .get(&req_input.req_id)
                    .and_then(|p| p.prompt_logprobs)
                    .is_some())
                || !req_input.spec_token_ids.is_empty();

            let (full_logits_f32, prompt_logprobs_info) = if needs_full_logits {
                let top_n = self
                    .sampling_params_map
                    .get(&req_input.req_id)
                    .and_then(|p| p.prompt_logprobs)
                    .map(|n| n.max(0) as usize);
                let f32_logits = logits.as_dtype(Dtype::Float32).map_err(|e| {
                    ExecutorError::WorkerExecution(format!("dtype cast error: {e}"))
                })?;
                let info = top_n.map(|n| (n, req_input.token_ids.clone()));
                (Some(f32_logits), info)
            } else {
                (None, None)
            };

            lazy_outputs[idx] = LazyReqOutput {
                last_logits,
                full_logits_f32,
                prompt_logprobs_info,
            };
        }

        // --- Phase B: Classify requests BEFORE eval ---
        // Classification only checks sampling_params, grammar_states, and spec_token_ids —
        // none of which depend on logit values. Safe to run on lazy (unevaluated) arrays.
        let mut gpu_greedy: Vec<usize> = Vec::new();
        let mut gpu_temp: Vec<(usize, f32)> = Vec::new();
        let mut cpu_fallback: Vec<usize> = Vec::new();

        for (req_idx, req_input) in req_inputs.iter().enumerate() {
            let lazy_out = &lazy_outputs[req_idx];

            // Compute prompt logprobs if requested (arrays are already materialized).
            if let Some((top_n, ref token_ids)) = lazy_out.prompt_logprobs_info {
                let full_logits_f32 = lazy_out.full_logits_f32.as_ref().unwrap();
                let num_positions = token_ids.len() - 1;
                let vocab_size = full_logits_f32.dim(-1) as usize;
                let flat = full_logits_f32.as_slice::<f32>();
                let mut plps = Vec::with_capacity(num_positions);
                for i in 0..num_positions {
                    let row = &flat[i * vocab_size..(i + 1) * vocab_size];
                    let actual_token = token_ids[i + 1];
                    plps.push(vllm_model::sampler::compute_logprobs(
                        row,
                        actual_token,
                        top_n,
                    ));
                }
                prompt_logprobs_map.insert(req_input.req_id.clone(), plps);
            }

            // Classify: speculative → CPU, grammar/penalties → CPU, else GPU.
            if !req_input.spec_token_ids.is_empty() {
                cpu_fallback.push(req_idx);
                continue;
            }

            #[cfg(feature = "guided-decoding")]
            let has_grammar = self
                .grammar_states
                .get_mut(&req_input.req_id)
                .is_some_and(|g| g.allowed_tokens().is_some());
            #[cfg(not(feature = "guided-decoding"))]
            let has_grammar = false;

            let params = self.sampling_params_map.get(&req_input.req_id);
            let needs_cpu = has_grammar
                || params.is_some_and(|p| {
                    p.repetition_penalty != 1.0
                        || p.frequency_penalty != 0.0
                        || p.presence_penalty != 0.0
                        || p.min_p > 0.0
                        || p.logit_bias.is_some()
                        || p.logprobs.is_some()
                        || (p.top_k > 0 || p.top_p < 1.0)
                });

            if needs_cpu {
                cpu_fallback.push(req_idx);
            } else if let Some(p) = params {
                let temp = p.temperature as f32;
                if temp < 1e-5 {
                    gpu_greedy.push(req_idx);
                } else {
                    gpu_temp.push((req_idx, temp));
                }
            } else {
                gpu_greedy.push(req_idx);
            }
        }

        // --- Phase C: Build lazy sampling ops (extend compute graph, no Metal work yet) ---
        // For gpu_greedy: lazy argmax on lazy logits.
        // For gpu_temp: lazy divide + categorical on lazy logits.
        // These extend the forward graph so everything fuses into one Metal command buffer.

        let gpu_greedy_result: Option<Array> = if gpu_greedy.len() > 1 {
            let logit_refs: Vec<Array> = gpu_greedy
                .iter()
                .map(|&i| lazy_outputs[i].last_logits.clone())
                .collect();
            let stacked = mlx_rs::ops::stack_axis(&logit_refs, 0)
                .map_err(|e| ExecutorError::WorkerExecution(format!("stack error: {e}")))?;
            Some(
                mlx_rs::ops::indexing::argmax_axis(&stacked, -1, None)
                    .map_err(|e| ExecutorError::WorkerExecution(format!("argmax error: {e}")))?,
            )
        } else if gpu_greedy.len() == 1 {
            Some(
                mlx_rs::ops::indexing::argmax_axis(
                    &lazy_outputs[gpu_greedy[0]].last_logits,
                    -1,
                    None,
                )
                .map_err(|e| ExecutorError::WorkerExecution(format!("argmax error: {e}")))?,
            )
        } else {
            None
        };

        // Build lazy temp sampling ops, grouped by temperature.
        let mut gpu_temp_results: Vec<(Vec<usize>, Array)> = Vec::new();
        if !gpu_temp.is_empty() {
            let mut temp_groups: HashMap<i32, Vec<usize>> = HashMap::new();
            for &(req_idx, temp) in &gpu_temp {
                let key = (temp * 100.0).round() as i32;
                temp_groups.entry(key).or_default().push(req_idx);
            }

            for (temp_key, group) in &temp_groups {
                let temp = *temp_key as f32 / 100.0;
                let logit_refs: Vec<Array> = group
                    .iter()
                    .map(|&i| lazy_outputs[i].last_logits.clone())
                    .collect();
                let logits = if logit_refs.len() > 1 {
                    mlx_rs::ops::stack_axis(&logit_refs, 0)
                        .map_err(|e| ExecutorError::WorkerExecution(format!("stack error: {e}")))?
                } else {
                    logit_refs.into_iter().next().unwrap()
                };
                let sampled = if temp < 1e-5 {
                    mlx_rs::ops::indexing::argmax_axis(&logits, -1, None)
                        .map_err(|e| ExecutorError::WorkerExecution(format!("argmax error: {e}")))?
                } else {
                    let scaled = logits
                        .as_dtype(Dtype::Float32)
                        .and_then(|s| s.divide(Array::from_f32(temp)))
                        .map_err(|e| ExecutorError::WorkerExecution(format!("scale error: {e}")))?;
                    mlx_rs::random::categorical(&scaled, None, None, None).map_err(|e| {
                        ExecutorError::WorkerExecution(format!("categorical error: {e}"))
                    })?
                };
                gpu_temp_results.push((group.clone(), sampled));
            }
        }

        // --- Phase D: Submit GPU work ---
        //
        // Double-buffering (mlx-lm _next() pattern): when we can defer
        // (no cpu_fallback, no prompt_logprobs) AND have a previous output
        // to return, use async_eval + store pending + return previous.
        // Otherwise fall back to synchronous eval.
        let can_defer = cpu_fallback.is_empty()
            && !any_logprobs_requested
            && lazy_outputs
                .iter()
                .all(|o| o.prompt_logprobs_info.is_none())
            && prev_output.is_some();

        if can_defer {
            // async_eval: submit GPU work, don't wait.
            let mut arrays_to_eval: Vec<&Array> = Vec::new();
            if let Some(ref arr) = gpu_greedy_result {
                arrays_to_eval.push(arr);
            }
            for (_, arr) in &gpu_temp_results {
                arrays_to_eval.push(arr);
            }
            mlx_rs::transforms::async_eval(arrays_to_eval)
                .map_err(|e| ExecutorError::WorkerExecution(format!("async_eval error: {e}")))?;

            // Store unread GPU arrays as pending for next call.
            let greedy_mapping: Vec<(String, usize)> = gpu_greedy
                .iter()
                .enumerate()
                .map(|(batch_pos, &req_idx)| (req_inputs[req_idx].req_id.clone(), batch_pos))
                .collect();
            let temp_pending: Vec<(Vec<(String, usize)>, Array)> = gpu_temp_results
                .into_iter()
                .map(|(group, arr)| {
                    let mapping: Vec<(String, usize)> = group
                        .iter()
                        .enumerate()
                        .map(|(batch_pos, &req_idx)| {
                            (req_inputs[req_idx].req_id.clone(), batch_pos)
                        })
                        .collect();
                    (mapping, arr)
                })
                .collect();

            self.pending_step = Some(PendingStep {
                req_ids: req_inputs.iter().map(|r| r.req_id.clone()).collect(),
                greedy_result: gpu_greedy_result,
                greedy_mapping,
                temp_results: temp_pending,
            });

            // Timing.
            let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
            let num_prefills = req_inputs.iter().filter(|r| r.is_prefill).count();
            let num_decodes = req_inputs.len() - num_prefills;
            self.step_count += req_inputs.len();
            if num_prefills > 0 {
                self.prefill_count += num_prefills;
                self.total_prefill_ms += step_ms * (num_prefills as f64 / req_inputs.len() as f64);
            }
            if num_decodes > 0 {
                self.decode_count += num_decodes;
                self.total_decode_ms += step_ms * (num_decodes as f64 / req_inputs.len() as f64);
            }

            // Return PREVIOUS step's output.  Current step's output will
            // be read and returned at the start of the next execute_model.
            return Ok(prev_output.unwrap());
        }

        // --- Synchronous fallback ---
        {
            let mut arrays_to_eval: Vec<&Array> = Vec::new();
            for &idx in &cpu_fallback {
                arrays_to_eval.push(&lazy_outputs[idx].last_logits);
            }
            for out in &lazy_outputs {
                if let Some(ref f32_logits) = out.full_logits_f32 {
                    arrays_to_eval.push(f32_logits);
                }
            }
            if let Some(ref arr) = gpu_greedy_result {
                arrays_to_eval.push(arr);
            }
            for (_, arr) in &gpu_temp_results {
                arrays_to_eval.push(arr);
            }
            mlx_rs::transforms::eval(arrays_to_eval)
                .map_err(|e| ExecutorError::WorkerExecution(format!("batch eval error: {e}")))?;
        }

        // --- Phase E: Read materialized results ---

        // Prompt logprobs (needs materialized full_logits_f32).
        for (req_idx, req_input) in req_inputs.iter().enumerate() {
            let lazy_out = &lazy_outputs[req_idx];
            if let Some((top_n, ref token_ids)) = lazy_out.prompt_logprobs_info {
                let full_logits_f32 = lazy_out.full_logits_f32.as_ref().unwrap();
                let num_positions = token_ids.len() - 1;
                let vocab_size = full_logits_f32.dim(-1) as usize;
                let flat = full_logits_f32.as_slice::<f32>();
                let mut plps = Vec::with_capacity(num_positions);
                for i in 0..num_positions {
                    let row = &flat[i * vocab_size..(i + 1) * vocab_size];
                    let actual_token = token_ids[i + 1];
                    plps.push(vllm_model::sampler::compute_logprobs(
                        row,
                        actual_token,
                        top_n,
                    ));
                }
                prompt_logprobs_map.insert(req_input.req_id.clone(), plps);
            }
        }

        // GPU greedy: read from materialized argmax result.
        if let Some(ref greedy_result) = gpu_greedy_result {
            let flat = greedy_result.as_slice::<u32>();
            for (batch_pos, &req_idx) in gpu_greedy.iter().enumerate() {
                token_map.insert(req_inputs[req_idx].req_id.clone(), vec![flat[batch_pos]]);
            }
        }

        // GPU temperature: read from materialized categorical results.
        for (group, sampled) in &gpu_temp_results {
            let flat = sampled.as_slice::<u32>();
            for (batch_pos, &req_idx) in group.iter().enumerate() {
                token_map.insert(req_inputs[req_idx].req_id.clone(), vec![flat[batch_pos]]);
            }
        }

        // CPU fallback: per-request sampling on materialized logits.
        for &req_idx in &cpu_fallback {
            let req_input = &req_inputs[req_idx];
            let lazy_out = &lazy_outputs[req_idx];

            let sampled = if !req_input.spec_token_ids.is_empty() {
                // Speculative decode verification (greedy).
                let all_logits_f32 = lazy_out.full_logits_f32.as_ref().unwrap();

                let vocab_size = all_logits_f32.dim(-1) as usize;
                let flat = all_logits_f32.as_slice::<f32>();
                let num_drafts = req_input.spec_token_ids.len();
                let num_positions = req_input.token_ids.len();

                let mut accepted = Vec::new();
                for i in 0..num_drafts {
                    if i >= num_positions {
                        break;
                    }
                    let row = &flat[i * vocab_size..(i + 1) * vocab_size];
                    let argmax_id = row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(0);

                    let draft = req_input.spec_token_ids[i];
                    if argmax_id == draft {
                        accepted.push(draft);
                    } else {
                        accepted.push(argmax_id);
                        break;
                    }
                }

                // Bonus token if all drafts accepted.
                if accepted.len() == num_drafts && num_positions > 0 {
                    let last_row_start = (num_positions - 1) * vocab_size;
                    let last_row = &flat[last_row_start..last_row_start + vocab_size];
                    let bonus = last_row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(0);
                    accepted.push(bonus);
                }

                if accepted.is_empty() {
                    let last_row_start = (num_positions - 1) * vocab_size;
                    let last_row = &flat[last_row_start..last_row_start + vocab_size];
                    let argmax = last_row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(0);
                    vec![argmax]
                } else {
                    debug!(
                        "Spec decode {}: {}/{} drafts accepted (+ {} tokens total)",
                        req_input.req_id,
                        accepted.len().min(num_drafts),
                        num_drafts,
                        accepted.len()
                    );
                    accepted
                }
            } else {
                // CPU sampling with penalties/grammar/logprobs.
                #[cfg(feature = "guided-decoding")]
                let grammar_allowed: Option<Vec<u32>> = self
                    .grammar_states
                    .get_mut(&req_input.req_id)
                    .and_then(|g| g.allowed_tokens());
                #[cfg(not(feature = "guided-decoding"))]
                let grammar_allowed: Option<Vec<u32>> = None;

                let flat = lazy_out.last_logits.as_slice::<f32>();

                let p = self.sampling_params_map.get(&req_input.req_id).unwrap();
                let prev_tokens = self
                    .token_buffers
                    .get(&req_input.req_id)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let (token_id, maybe_logprobs) =
                    cpu_sampler.sample_one(flat, p, prev_tokens, grammar_allowed.as_deref());
                if let Some(lp) = maybe_logprobs {
                    logprobs_map
                        .entry(req_input.req_id.clone())
                        .or_default()
                        .push(lp);
                }
                vec![token_id]
            };

            token_map.insert(req_input.req_id.clone(), sampled);
        }

        // --- Post-sampling: update grammar states and token buffers ---
        for (req_idx, req_input) in req_inputs.iter().enumerate() {
            let sampled = token_map.get(&req_input.req_id).unwrap();

            // Advance grammar state with the sampled token.
            #[cfg(feature = "guided-decoding")]
            if let Some(guide) = self.grammar_states.get_mut(&req_input.req_id)
                && let Some(&token_id) = sampled.first()
            {
                guide.advance(token_id);
            }
            let _ = req_idx; // suppress unused warning

            // Update token buffer.
            if let Some(buf) = self.token_buffers.get_mut(&req_input.req_id) {
                buf.extend_from_slice(sampled);
            }
        }

        // Timing bookkeeping (aggregate for all requests in this batch).
        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        let num_prefills = req_inputs.iter().filter(|r| r.is_prefill).count();
        let num_decodes = req_inputs.len() - num_prefills;
        self.step_count += req_inputs.len();
        if num_prefills > 0 {
            self.prefill_count += num_prefills;
            self.total_prefill_ms += step_ms * (num_prefills as f64 / req_inputs.len() as f64);
        }
        if num_decodes > 0 {
            self.decode_count += num_decodes;
            self.total_decode_ms += step_ms * (num_decodes as f64 / req_inputs.len() as f64);
        }

        // Build ModelRunnerOutput with logprobs if any were collected.
        let mut output = ModelRunnerOutput::from_token_map(token_map);
        if any_logprobs_requested && !logprobs_map.is_empty() {
            let logprobs_vec: Vec<Option<Vec<vllm_common::LogprobsOutput>>> = output
                .req_ids
                .iter()
                .map(|rid| logprobs_map.remove(rid))
                .collect();
            output.logprobs = Some(logprobs_vec);
        }
        output.prompt_logprobs_dict = prompt_logprobs_map;
        Ok(output)
    }

    fn embed(
        &mut self,
        token_id_seqs: &[&[u32]],
    ) -> vllm_executor::error::ExecutorResult<Vec<Vec<f32>>> {
        use vllm_executor::error::ExecutorError;
        use vllm_model::embedding::PoolingStrategy;

        let strategy = self.pooling_strategy;
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| ExecutorError::WorkerExecution("model not loaded".into()))?;

        let mut results = Vec::with_capacity(token_id_seqs.len());
        for token_ids in token_id_seqs {
            let input_ids = Array::from_iter(
                token_ids.iter().map(|&t| t as i32),
                &[token_ids.len() as i32],
            );
            let positions = Array::from_iter(
                (0..token_ids.len() as i32).collect::<Vec<_>>(),
                &[token_ids.len() as i32],
            );

            let hidden_states = model
                .hidden_states(&input_ids, &positions)
                .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

            // Pool according to strategy.
            let num_tokens = token_ids.len() as i32;
            let pooled = match strategy {
                PoolingStrategy::Last => hidden_states.index(num_tokens - 1),
                PoolingStrategy::Cls => hidden_states.index(0),
                PoolingStrategy::Mean => {
                    // Average across the token dimension (axis 0).
                    let sum = hidden_states
                        .sum_axis(0, None)
                        .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
                    sum.divide(Array::from(num_tokens as f32))
                        .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?
                }
            };

            // L2 normalize: compute norm, divide, cast to f32.
            let sq = pooled
                .square()
                .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
            let sum = sq
                .sum(None)
                .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
            let norm = sum
                .sqrt()
                .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
            let normalized = pooled
                .divide(&norm)
                .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
            let normalized_f32 = normalized
                .as_dtype(mlx_rs::Dtype::Float32)
                .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

            // Evaluate and convert to Vec<f32>.
            normalized_f32
                .eval()
                .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
            let vec: Vec<f32> = normalized_f32.as_slice().to_vec();
            results.push(vec);
        }
        Ok(results)
    }

    fn take_preloaded_tokenizer(&mut self) -> Option<tokenizers::Tokenizer> {
        self.preloaded_tokenizer.take()
    }

    fn architecture(&self) -> Option<String> {
        self.resolved_architecture.clone()
    }

    fn shutdown(&mut self) {
        self.pending_step = None;
        self.is_shutdown = true;
        self.model = None;
        info!("MlxWorker: shut down");
    }

    fn rank(&self) -> usize {
        0
    }

    fn local_rank(&self) -> usize {
        0
    }

    fn is_driver_worker(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_worker() -> MlxWorker {
        MlxWorker::new(MlxWorkerConfig {
            model_path: "/nonexistent".to_string(),
            dtype: "f16".to_string(),
            hf_token: None,
            cache_dir: None,
            block_size: 16,
            lora_adapter: None,
            pooling_strategy: "auto".to_string(),
            is_pooling: false,
            enable_prefix_caching: false,
        })
    }

    #[test]
    fn test_timing_fields_initial() {
        let w = make_worker();
        assert_eq!(w.step_count(), 0);
        assert_eq!(w.prefill_count(), 0);
        assert_eq!(w.decode_count(), 0);
        assert_eq!(w.avg_prefill_ms(), 0.0);
        assert_eq!(w.avg_decode_ms(), 0.0);
    }

    #[test]
    fn test_avg_latency_zero_division() {
        // avg should return 0.0 when no steps have been run, not panic.
        let w = make_worker();
        assert_eq!(w.avg_prefill_ms(), 0.0);
        assert_eq!(w.avg_decode_ms(), 0.0);
    }

    #[test]
    fn test_execute_model_no_model_loaded() {
        // Without a loaded model, execute_model returns an error.
        let mut w = make_worker();
        let sched = SchedulerOutput::make_empty();
        let result = w.execute_model(&sched);
        assert!(result.is_err());
        // Timing counters should not be updated on error.
        assert_eq!(w.step_count(), 0);
    }

    #[test]
    fn test_mlx_worker_config_dtype_parsing() {
        let config = MlxWorkerConfig {
            model_path: String::new(),
            dtype: "auto".to_string(),
            hf_token: None,
            cache_dir: None,
            block_size: 16,
            lora_adapter: None,
            pooling_strategy: "auto".to_string(),
            is_pooling: false,
            enable_prefix_caching: false,
        };
        assert!(config.mlx_dtype().unwrap().is_none());

        let config_f16 = MlxWorkerConfig {
            dtype: "f16".to_string(),
            ..config.clone()
        };
        assert_eq!(config_f16.mlx_dtype().unwrap(), Some(Dtype::Float16));

        let config_bad = MlxWorkerConfig {
            dtype: "int8".to_string(),
            ..config
        };
        assert!(config_bad.mlx_dtype().is_err());
    }
}
