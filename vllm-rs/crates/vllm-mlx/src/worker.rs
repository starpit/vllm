// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! MLX worker implementing the `Worker` trait from `vllm-executor`.
//!
//! Key difference from CandleWorker: all model ops are lazy. The forward pass
//! builds a compute graph, and a single `eval()` call materializes the entire
//! graph as minimal Metal command buffers. This eliminates the ~42ms of
//! per-dispatch overhead seen with candle's eager execution on Metal.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};
use tracing::{info, warn};

use vllm_common::SamplingParams;
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_engine::executor::ModelRunnerOutput;
use vllm_executor::error::{ExecutorError, ExecutorResult};
use vllm_executor::worker::Worker;
use vllm_model::weight::HfModelConfig;

use crate::cache::{self, MlxKvCache};
use crate::models::{MlxModel, MlxModelRegistry};

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

    /// Per-request grammar guide state for constrained decoding.
    grammar_states: HashMap<String, vllm_models::grammar::GrammarGuide>,
    /// Compiled vocabulary for grammar-guided decoding (built once from tokenizer).
    grammar_vocabulary: Option<outlines_core::vocabulary::Vocabulary>,

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
            grammar_states: HashMap::new(),
            grammar_vocabulary: None,
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

    /// Build grammar vocabulary on demand (lazy — deferred from startup).
    fn ensure_grammar_vocabulary(&mut self) {
        if self.grammar_vocabulary.is_some() {
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
        let tokenizer = match tokenizers::Tokenizer::from_file(&tokenizer_path) {
            Ok(t) => t,
            Err(e) => {
                warn!("MlxWorker: failed to load tokenizer.json for grammar vocabulary: {e}");
                return;
            }
        };

        let hf_vocab = tokenizer.get_vocab(true);
        let eos_token_id = tokenizer.token_to_id("</s>").unwrap_or(0);
        let tokens: Vec<(u32, String)> = hf_vocab.into_iter().map(|(s, id)| (id, s)).collect();

        match vllm_models::grammar::build_vocabulary(&tokens, eos_token_id) {
            Ok(vocab) => {
                info!(
                    "MlxWorker: grammar vocabulary built ({} tokens, eos={})",
                    vocab.len(),
                    eos_token_id
                );
                self.grammar_vocabulary = Some(vocab);
            }
            Err(e) => {
                warn!("MlxWorker: failed to build grammar vocabulary: {e}");
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

    /// Greedy decode: take argmax of logits (CPU side, after eval).
    fn greedy_sample(logits: &Array) -> ExecutorResult<Vec<u32>> {
        let indices = mlx_rs::ops::indexing::argmax_axis(logits, -1, None)
            .map_err(|e| ExecutorError::WorkerExecution(format!("argmax error: {e}")))?;
        indices
            .eval()
            .map_err(|e| ExecutorError::WorkerExecution(format!("eval error: {e}")))?;

        // Extract as u32 values.
        let flat = indices.as_slice::<u32>();
        Ok(flat.to_vec())
    }

    /// Sample with temperature from logits.
    fn sample_with_temperature(logits: &Array, temperature: f32) -> ExecutorResult<Vec<u32>> {
        if temperature < 1e-5 {
            return Self::greedy_sample(logits);
        }

        // Scale logits by temperature.
        let logits_f32 = logits
            .as_dtype(Dtype::Float32)
            .map_err(|e| ExecutorError::WorkerExecution(format!("dtype cast error: {e}")))?;
        let scaled = logits_f32
            .divide(Array::from_f32(temperature))
            .map_err(|e| ExecutorError::WorkerExecution(format!("scale error: {e}")))?;

        // categorical() expects logits and applies softmax internally.
        let sampled = mlx_rs::random::categorical(&scaled, None, None, None)
            .map_err(|e| ExecutorError::WorkerExecution(format!("categorical error: {e}")))?;

        sampled
            .eval()
            .map_err(|e| ExecutorError::WorkerExecution(format!("eval error: {e}")))?;

        let flat = sampled.as_slice::<u32>();
        Ok(flat.to_vec())
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

        // Look up architecture in the MLX registry.
        let arch = hf_config
            .architectures
            .first()
            .ok_or_else(|| {
                ExecutorError::WorkerInit("config.json has no architectures field".to_string())
            })?
            .clone();
        let registry = MlxModelRegistry::default_registry();
        let factory = registry.get_factory(&arch, is_quantized).ok_or_else(|| {
            ExecutorError::WorkerInit(format!(
                "unsupported MLX architecture: {arch}. Supported: {:?}",
                registry.architectures().collect::<Vec<_>>()
            ))
        })?;

        // Load the model.
        let model = factory(&model_dir, &hf_config, dtype).map_err(|e| {
            ExecutorError::WorkerInit(format!("failed to construct MLX model: {e}"))
        })?;

        self.model_dir = Some(model_dir);
        self.hf_config = Some(hf_config);
        self.resolved_dtype = Some(dtype);
        self.model = Some(model);
        let quant_str = if is_quantized { ", quantized" } else { "" };
        info!("MlxWorker: model loaded (arch={arch}, dtype={dtype:?}{quant_str})");

        // Grammar vocabulary is built lazily on first constrained-decoding request.

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
        use sysinfo::System;
        let sys = System::new_with_specifics(
            sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
        );
        let total = sys.total_memory() as usize;
        if total == 0 {
            return Ok(8 * 1024 * 1024 * 1024);
        }
        Ok(total)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        // Lazily build grammar vocabulary if any new request needs constrained decoding.
        // Done before borrowing self.model to satisfy the borrow checker.
        let needs_grammar = scheduler_output.scheduled_new_reqs.iter().any(|r| {
            r.sampling_params
                .as_ref()
                .is_some_and(|p| p.guided_grammar.is_some())
        });
        if needs_grammar {
            self.ensure_grammar_vocabulary();
        }

        let model = self
            .model
            .as_mut()
            .ok_or_else(|| ExecutorError::WorkerExecution("model not loaded".to_string()))?;

        let num_layers = model.num_layers();

        // Clean up finished requests.
        for req_id in &scheduler_output.finished_req_ids {
            self.token_buffers.remove(req_id);
            self.sampling_params_map.remove(req_id);
            self.kv_caches.remove(req_id);
            self.grammar_states.remove(req_id);
        }

        // Collect requests.
        struct ReqInput {
            req_id: String,
            token_ids: Vec<u32>,
            positions: Vec<i32>,
            is_prefill: bool,
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
            let tokens_to_use = &prompt_ids[..num_tokens.min(prompt_ids.len())];

            self.token_buffers
                .insert(new_req.req_id.clone(), prompt_ids.to_vec());
            if let Some(ref params) = new_req.sampling_params {
                // Create grammar guide for constrained decoding if requested.
                if let Some(ref grammar) = params.guided_grammar {
                    if let Some(ref vocab) = self.grammar_vocabulary {
                        match vllm_models::grammar::GrammarGuide::from_guided_grammar(
                            grammar, vocab,
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
            self.kv_caches
                .insert(new_req.req_id.clone(), cache::empty_kv_cache(num_layers));

            let pos_offset = new_req.num_computed_tokens as i32;
            let positions: Vec<i32> = (0..tokens_to_use.len() as i32)
                .map(|i| pos_offset + i)
                .collect();

            req_inputs.push(ReqInput {
                req_id: new_req.req_id.clone(),
                token_ids: tokens_to_use.to_vec(),
                positions,
                is_prefill: true,
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
                req_inputs.push(ReqInput {
                    req_id: req_id.clone(),
                    token_ids: vec![last_token],
                    positions: vec![position],
                    is_prefill: false,
                });
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
                });
            }
        }

        if req_inputs.is_empty() {
            return Ok(ModelRunnerOutput::from_token_map(HashMap::new()));
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
        let mut cpu_sampler = vllm_models::Sampler::new();

        // Run a separate forward pass per request (each has its own KV cache).
        for req_input in &req_inputs {
            let step_start = Instant::now();

            let input_ids = Array::from_iter(
                req_input.token_ids.iter().map(|&t| t as i32),
                &[req_input.token_ids.len() as i32],
            );
            let positions = Array::from_iter(
                req_input.positions.iter().copied(),
                &[req_input.positions.len() as i32],
            );

            // Get or create KV cache for this request.
            let kv_cache = self
                .kv_caches
                .entry(req_input.req_id.clone())
                .or_insert_with(|| cache::empty_kv_cache(num_layers));

            // Forward pass — builds lazy compute graph (no eval yet).
            let logits = model
                .forward(&input_ids, &positions, kv_cache)
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("model forward failed: {e}"))
                })?;

            // Compute prompt logprobs if requested and this is a prefill.
            if req_input.is_prefill
                && req_input.token_ids.len() > 1
                && let Some(params) = self.sampling_params_map.get(&req_input.req_id)
                && let Some(top_n) = params.prompt_logprobs
            {
                let top_n = top_n.max(0) as usize;
                let num_positions = req_input.token_ids.len() - 1;
                // Eval the full logits to extract prompt-position logits.
                logits
                    .eval()
                    .map_err(|e| ExecutorError::WorkerExecution(format!("eval error: {e}")))?;
                let full_logits_f32 = logits.as_dtype(Dtype::Float32).map_err(|e| {
                    ExecutorError::WorkerExecution(format!("dtype cast error: {e}"))
                })?;
                full_logits_f32
                    .eval()
                    .map_err(|e| ExecutorError::WorkerExecution(format!("eval error: {e}")))?;
                // full_logits_f32 shape: [num_tokens, vocab_size]
                let vocab_size = full_logits_f32.dim(-1) as usize;
                let flat = full_logits_f32.as_slice::<f32>();
                let mut plps = Vec::with_capacity(num_positions);
                for i in 0..num_positions {
                    let row = &flat[i * vocab_size..(i + 1) * vocab_size];
                    let actual_token = req_input.token_ids[i + 1];
                    plps.push(vllm_models::sampler::compute_logprobs(
                        row,
                        actual_token,
                        top_n,
                    ));
                }
                prompt_logprobs_map.insert(req_input.req_id.clone(), plps);
            }

            // Take logits at the last position (still lazy — no eval).
            let last_pos = req_input.token_ids.len() - 1;
            let req_logits = if req_input.token_ids.len() > 1 {
                logits
                    .index(last_pos as i32)
                    .expand_dims(0)
                    .map_err(|e| ExecutorError::WorkerExecution(format!("index error: {e}")))?
            } else {
                logits
            };

            // Query grammar-allowed tokens if constrained decoding is active.
            let grammar_allowed: Option<Vec<u32>> = self
                .grammar_states
                .get(&req_input.req_id)
                .and_then(|g| g.allowed_tokens());
            let has_grammar = grammar_allowed.is_some();

            // Determine if we need CPU-side sampling (penalties/logprobs/grammar/etc.).
            let params = self.sampling_params_map.get(&req_input.req_id);
            let needs_cpu_sampling = has_grammar
                || params.is_some_and(|p| {
                    p.repetition_penalty != 1.0
                        || p.frequency_penalty != 0.0
                        || p.presence_penalty != 0.0
                        || p.min_p > 0.0
                        || p.logit_bias.is_some()
                        || p.logprobs.is_some()
                        || (p.top_k > 0 || p.top_p < 1.0)
                });

            let sampled = if needs_cpu_sampling {
                // Eval the logits graph, extract to CPU Vec<f32>, and use
                // the unified CPU sampler which handles all penalty/filter/logprobs.
                req_logits
                    .eval()
                    .map_err(|e| ExecutorError::WorkerExecution(format!("eval error: {e}")))?;
                let logits_f32 = req_logits.as_dtype(Dtype::Float32).map_err(|e| {
                    ExecutorError::WorkerExecution(format!("dtype cast error: {e}"))
                })?;
                logits_f32
                    .eval()
                    .map_err(|e| ExecutorError::WorkerExecution(format!("eval error: {e}")))?;
                let flat = logits_f32.as_slice::<f32>();

                let p = params.unwrap();
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
            } else if let Some(p) = params {
                // Simple temperature sampling via MLX (no penalties needed).
                let temp = p.temperature as f32;
                Self::sample_with_temperature(&req_logits, temp)?
            } else {
                Self::greedy_sample(&req_logits)?
            };

            let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

            // Advance grammar state with the sampled token.
            if let Some(guide) = self.grammar_states.get_mut(&req_input.req_id)
                && let Some(&token_id) = sampled.first()
            {
                guide.advance(token_id);
            }

            // Update token buffer.
            if let Some(buf) = self.token_buffers.get_mut(&req_input.req_id) {
                buf.extend_from_slice(&sampled);
            }

            // Timing bookkeeping.
            self.step_count += 1;
            if req_input.is_prefill {
                self.prefill_count += 1;
                self.total_prefill_ms += step_ms;
            } else {
                self.decode_count += 1;
                self.total_decode_ms += step_ms;
            }

            token_map.insert(req_input.req_id.clone(), sampled);
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

    fn shutdown(&mut self) {
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
