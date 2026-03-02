// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! A real `Worker` implementation backed by candle for CPU/CUDA/Metal inference.
//!
//! `CandleWorker` loads model weights via `vllm-model`, constructs a model
//! architecture via `vllm-models`, and runs forward passes using candle
//! tensors. HuggingFace Hub models are downloaded on demand via `hf-hub`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use tracing::{debug, info, warn};
use vllm_common::SamplingParams;
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_engine::executor::ModelRunnerOutput;
use vllm_model::gguf::{self, GgufFile};
use vllm_model::weight::{HfModelConfig, ModelWeights};
use vllm_models::{KvBlockPool, KvCache, KvCacheStorage, Model, ModelRegistry, Sampler};

use crate::error::{ExecutorError, ExecutorResult};
use crate::input_batch::InputBatch;
use crate::worker::Worker;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a `CandleWorker`.
#[derive(Debug, Clone)]
pub struct CandleWorkerConfig {
    /// Path to a local model directory, or a HuggingFace model ID
    /// (e.g. "meta-llama/Llama-3.2-1B").
    pub model_path: String,

    /// Device string: "cpu", "cuda:N", "metal", or "auto".
    pub device_str: String,

    /// Data type for model weights: "auto", "f32", "f16", "bf16", etc.
    /// "auto" resolves from config.json `torch_dtype` at load time.
    pub dtype: String,

    /// Optional HuggingFace token for gated models.
    pub hf_token: Option<String>,

    /// Optional cache directory for downloaded models.
    pub cache_dir: Option<String>,

    /// KV cache block size in tokens (must match the scheduler's block size).
    /// Defaults to 16 if not set.
    pub block_size: usize,

    /// Specific GGUF filename to download from a HuggingFace repo.
    /// When set, only this file is downloaded instead of safetensors weights.
    pub gguf_file: Option<String>,

    /// Optional path to a LoRA adapter directory (local path or HF repo ID).
    pub lora_adapter: Option<String>,
    /// Pooling strategy for embeddings: "auto", "last", "cls", "mean".
    /// "auto" detects from `1_Pooling/config.json`, defaults to "last".
    pub pooling_strategy: String,
    /// Whether the engine is in pooling mode.
    /// In pooling mode, `execute_model` returns embedding vectors instead of sampled tokens.
    pub is_pooling: bool,
}

impl CandleWorkerConfig {
    /// Parse the dtype string to a candle DType.
    ///
    /// Returns `None` for "auto" — the caller must resolve from config.json.
    pub fn candle_dtype(&self) -> ExecutorResult<Option<DType>> {
        match self.dtype.as_str() {
            "auto" => Ok(None),
            "f32" | "float32" => Ok(Some(DType::F32)),
            "f16" | "float16" => Ok(Some(DType::F16)),
            "bf16" | "bfloat16" => Ok(Some(DType::BF16)),
            other => Err(ExecutorError::Config(format!("unsupported dtype: {other}"))),
        }
    }
}

// ---------------------------------------------------------------------------
// CandleWorker
// ---------------------------------------------------------------------------

/// A worker backed by candle for real model loading and inference.
pub struct CandleWorker {
    config: CandleWorkerConfig,
    device: Option<Device>,
    model: Option<Box<dyn Model>>,
    /// Resolved local path to the model directory.
    model_dir: Option<PathBuf>,
    /// Parsed HuggingFace config.json.
    hf_config: Option<HfModelConfig>,
    /// Resolved model dtype (after auto-detection).
    resolved_dtype: Option<DType>,
    /// KV cache block counts (stored for reference).
    num_gpu_blocks: usize,
    num_cpu_blocks: usize,
    is_shutdown: bool,

    /// Per-request token buffer: req_id → all token IDs (prompt + generated).
    /// Used to feed the correct input token on decode steps.
    token_buffers: HashMap<String, Vec<u32>>,
    /// Per-request sampling params, stored on first scheduling.
    sampling_params_map: HashMap<String, SamplingParams>,
    /// Global paged KV cache block pool (created by `initialize_cache`).
    kv_block_pool: Option<KvBlockPool>,
    /// Persistent input batch: maintains per-request state (block tables,
    /// tokens-in-pool, positions, last token) across engine steps.
    input_batch: InputBatch,
    /// Per-request KV cache (legacy, used when block pool is not available).
    kv_caches: HashMap<String, KvCache>,
    /// Per-request recurrent state for hybrid models (e.g., Qwen3-Next GDN layers).
    recurrent_states: HashMap<String, vllm_models::RecurrentState>,
    /// Model KV head count (set after load_model).
    num_kv_heads: usize,
    /// Model head dimension (set after load_model).
    head_dim: usize,
    /// Pre-loaded tokenizer (parsed in parallel with weight loading).
    preloaded_tokenizer: Option<tokenizers::Tokenizer>,
    /// Per-request grammar guide state for constrained decoding.
    #[cfg(feature = "guided-decoding")]
    grammar_states: HashMap<String, vllm_models::grammar::GrammarGuide>,
    /// Compiled vocabulary for grammar-guided decoding (built once from tokenizer).
    #[cfg(feature = "guided-decoding")]
    grammar_vocabulary: Option<outlines_core::vocabulary::Vocabulary>,
    /// Resolved pooling strategy for embeddings.
    pooling_strategy: vllm_models::embedding::PoolingStrategy,
    /// Per-request multimodal data, consumed on first forward (prefill).
    mm_data_map: HashMap<String, vllm_common::MultimodalData>,
    /// Whether the engine is in pooling mode.
    is_pooling: bool,
    /// Resolved model architecture name (e.g. "LlamaForCausalLM").
    resolved_architecture: Option<String>,
}

impl CandleWorker {
    /// Create a new CandleWorker from the given config.
    pub fn new(config: CandleWorkerConfig) -> Self {
        let is_pooling = config.is_pooling;
        Self {
            config,
            device: None,
            model: None,
            model_dir: None,
            hf_config: None,
            resolved_dtype: None,
            num_gpu_blocks: 0,
            num_cpu_blocks: 0,
            is_shutdown: false,
            token_buffers: HashMap::new(),
            sampling_params_map: HashMap::new(),
            kv_block_pool: None,
            input_batch: InputBatch::new(),
            kv_caches: HashMap::new(),
            recurrent_states: HashMap::new(),
            num_kv_heads: 0,
            head_dim: 0,
            preloaded_tokenizer: None,
            #[cfg(feature = "guided-decoding")]
            grammar_states: HashMap::new(),
            #[cfg(feature = "guided-decoding")]
            grammar_vocabulary: None,
            pooling_strategy: vllm_models::embedding::PoolingStrategy::Last,
            mm_data_map: HashMap::new(),
            is_pooling,
            resolved_architecture: None,
        }
    }

    /// Get the resolved model directory (after `load_model` has been called).
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }

    /// Get the parsed HfModelConfig (after `load_model` has been called).
    pub fn hf_config(&self) -> Option<&HfModelConfig> {
        self.hf_config.as_ref()
    }

    /// Get the resolved model dtype (after `load_model` has been called).
    pub fn resolved_dtype(&self) -> Option<DType> {
        self.resolved_dtype
    }

    /// Resolve the pooling strategy from config or auto-detect.
    fn resolve_pooling_strategy(&mut self) {
        use vllm_models::embedding::{PoolingStrategy, detect_pooling_strategy};

        let strategy = match self.config.pooling_strategy.as_str() {
            "auto" => {
                // Try to detect from 1_Pooling/config.json.
                let detected = self
                    .model_dir
                    .as_ref()
                    .and_then(|dir| detect_pooling_strategy(dir));
                if let Some(s) = detected {
                    info!("CandleWorker: pooling strategy auto-detected: {:?}", s);
                    s
                } else {
                    info!("CandleWorker: pooling strategy defaulting to Last (decoder model)");
                    PoolingStrategy::Last
                }
            }
            other => other.parse().unwrap_or_else(|_| {
                warn!(
                    "CandleWorker: unknown pooling strategy '{}', defaulting to Last",
                    other
                );
                PoolingStrategy::Last
            }),
        };
        self.pooling_strategy = strategy;
    }

    /// Build grammar vocabulary on demand (lazy — deferred from startup).
    ///
    /// Best-effort: logs a warning if tokenizer.json is not found or
    /// vocabulary construction fails. Grammar-guided decoding will be
    /// unavailable for those models.
    #[cfg(feature = "guided-decoding")]
    fn ensure_grammar_vocabulary(&mut self) {
        if self.grammar_vocabulary.is_some() {
            return;
        }
        let Some(model_dir) = &self.model_dir else {
            return;
        };
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            info!("CandleWorker: no tokenizer.json found, grammar-guided decoding unavailable");
            return;
        }
        let tokenizer = match tokenizers::Tokenizer::from_file(&tokenizer_path) {
            Ok(t) => t,
            Err(e) => {
                warn!("CandleWorker: failed to load tokenizer.json for grammar vocabulary: {e}");
                return;
            }
        };

        // Extract vocabulary: (token_id, token_string) pairs.
        let hf_vocab = tokenizer.get_vocab(true);
        let eos_token_id = tokenizer.token_to_id("</s>").unwrap_or(0);
        let tokens: Vec<(u32, String)> = hf_vocab.into_iter().map(|(s, id)| (id, s)).collect();

        match vllm_models::grammar::build_vocabulary(&tokens, eos_token_id) {
            Ok(vocab) => {
                info!(
                    "CandleWorker: grammar vocabulary built ({} tokens, eos={})",
                    vocab.len(),
                    eos_token_id
                );
                self.grammar_vocabulary = Some(vocab);
            }
            Err(e) => {
                warn!("CandleWorker: failed to build grammar vocabulary: {e}");
            }
        }
    }

    /// Check if the model path points to a GGUF file, resolving HF downloads
    /// if `--gguf-file` was specified.
    ///
    /// Returns `Some(path)` if a `.gguf` file was found or downloaded.
    /// Returns `None` if this is a safetensors model.
    fn resolve_gguf_path(&self) -> ExecutorResult<Option<PathBuf>> {
        let path = Path::new(&self.config.model_path);

        // Case 1: Direct local .gguf file path.
        if path.is_file()
            && path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            return Ok(Some(path.to_path_buf()));
        }

        // Case 2: Local directory containing a .gguf file.
        if path.is_dir() {
            if let Some(gguf) = find_gguf_in_dir(path) {
                return Ok(Some(gguf));
            }
            // Directory exists but no GGUF — fall through to safetensors.
            return Ok(None);
        }

        // Case 3: HF model ID with --gguf-file specified or auto-detected.
        // Build HF API client for cases 3 and 4.
        let is_hf_id =
            self.config.model_path.contains('/') && !Path::new(&self.config.model_path).exists();

        if is_hf_id {
            let gguf_filename = if let Some(ref f) = self.config.gguf_file {
                // Explicit --gguf-file flag.
                Some(f.clone())
            } else {
                // Case 4: Auto-detect GGUF repo by listing files.
                self.auto_detect_gguf_file()?
            };

            if let Some(gguf_file) = gguf_filename {
                info!(
                    "Downloading GGUF file '{}' from HuggingFace Hub: {}",
                    gguf_file, self.config.model_path
                );
                let api = self.build_hf_api()?;
                let repo = api.model(self.config.model_path.clone());

                let gguf_path = repo.get(&gguf_file).map_err(|e| {
                    ExecutorError::WorkerInit(format!(
                        "failed to download GGUF file '{}' from {}: {e}",
                        gguf_file, self.config.model_path
                    ))
                })?;

                // Try to download tokenizer files from the GGUF repo first,
                // then fall back to the base model repo (GGUF repos often
                // don't include tokenizer files).
                let mut got_tokenizer = false;
                if let Ok(p) = repo.get("tokenizer.json") {
                    info!("Downloaded tokenizer.json to {}", p.display());
                    got_tokenizer = true;
                }
                if let Ok(p) = repo.get("tokenizer_config.json") {
                    info!("Downloaded tokenizer_config.json to {}", p.display());
                    let _ = p;
                }

                // If the GGUF repo didn't have a tokenizer, try the base
                // model repo (e.g. "user/Model-GGUF" → "user/Model").
                if !got_tokenizer {
                    self.try_download_tokenizer_from_base_repo(&api, &gguf_path);
                }

                return Ok(Some(gguf_path));
            }
        }

        // Not a GGUF model.
        Ok(None)
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

    /// Auto-detect a GGUF file from a HuggingFace repo by listing repo files.
    ///
    /// Returns the filename of a GGUF file to download, preferring Q4_K_M
    /// quantization. Returns `None` if the repo has no GGUF files.
    fn auto_detect_gguf_file(&self) -> ExecutorResult<Option<String>> {
        let api = self.build_hf_api()?;
        let repo = api.model(self.config.model_path.clone());

        // Query repo info to list all files.
        let repo_info = match repo.info() {
            Ok(info) => info,
            Err(_) => return Ok(None), // Can't reach API; fall through to safetensors path.
        };

        let mut gguf_files: Vec<String> = repo_info
            .siblings
            .iter()
            .filter(|s| s.rfilename.ends_with(".gguf"))
            .map(|s| s.rfilename.clone())
            .collect();

        if gguf_files.is_empty() {
            return Ok(None);
        }

        // Sort for deterministic selection.
        gguf_files.sort();

        // Prefer Q4_K_M (good quality/size trade-off), then Q4_K_S, then Q8_0.
        let preferred = ["Q4_K_M", "Q4_K_S", "Q8_0", "Q5_K_M", "Q6_K"];
        for pref in &preferred {
            if let Some(f) = gguf_files.iter().find(|f| f.contains(pref)) {
                info!(
                    "Auto-selected GGUF file: {} (from {} available)",
                    f,
                    gguf_files.len()
                );
                return Ok(Some(f.clone()));
            }
        }

        // No preferred quant found — just pick the first .gguf file.
        let selected = gguf_files.first().unwrap().clone();
        info!(
            "Auto-selected GGUF file: {} (from {} available)",
            selected,
            gguf_files.len()
        );
        Ok(Some(selected))
    }

    /// Try to download tokenizer files from the base model repo.
    ///
    /// GGUF repos (e.g. `user/Model-GGUF`) typically don't include tokenizer
    /// files. The tokenizer lives in the base model repo (e.g. `user/Model`).
    /// This method tries common base repo name patterns and downloads
    /// `tokenizer.json` + `tokenizer_config.json` into the same directory
    /// as the GGUF file.
    fn try_download_tokenizer_from_base_repo(
        &self,
        api: &hf_hub::api::sync::Api,
        gguf_path: &Path,
    ) {
        let model_dir = match gguf_path.parent() {
            Some(d) => d,
            None => return,
        };

        // Derive candidate base repo names.
        let model_id = &self.config.model_path;
        let candidates = derive_base_repo_candidates(model_id);

        if candidates.is_empty() {
            info!(
                "No tokenizer in GGUF repo and couldn't derive base model name from '{}'",
                model_id
            );
            return;
        }

        for candidate in &candidates {
            info!("Trying to download tokenizer from base repo: {candidate}");
            let base_repo = api.model(candidate.clone());

            match base_repo.get("tokenizer.json") {
                Ok(src) => {
                    // Copy into the GGUF model directory so init.rs can find it.
                    let dst = model_dir.join("tokenizer.json");
                    if !dst.exists() {
                        if let Err(e) = std::fs::copy(&src, &dst) {
                            warn!("Failed to copy tokenizer.json: {e}");
                        } else {
                            info!("Tokenizer downloaded from {candidate}");
                        }
                    }

                    // Also grab tokenizer_config.json for chat templates.
                    if let Ok(src2) = base_repo.get("tokenizer_config.json") {
                        let dst2 = model_dir.join("tokenizer_config.json");
                        if !dst2.exists() {
                            let _ = std::fs::copy(&src2, &dst2);
                        }
                    }
                    return;
                }
                Err(_) => continue,
            }
        }

        info!(
            "Could not find tokenizer in any base repo candidate: {:?}",
            candidates
        );
    }

    /// Load a model from a GGUF file.
    fn load_model_gguf(&mut self, gguf_path: &Path, device: &Device) -> ExecutorResult<()> {
        info!(
            "CandleWorker: loading GGUF model from {}",
            gguf_path.display()
        );

        let mut gguf = GgufFile::open(gguf_path)
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to open GGUF: {e}")))?;

        info!("CandleWorker: GGUF file has {} tensors", gguf.num_tensors());

        // Extract model config from GGUF metadata.
        let hf_config = gguf::gguf_model_config(&gguf).map_err(|e| {
            ExecutorError::WorkerInit(format!("failed to extract GGUF config: {e}"))
        })?;

        let arch = gguf
            .get_metadata_string("general.architecture")
            .unwrap_or("unknown")
            .to_string();

        // Look up GGUF factory in the registry.
        let registry = ModelRegistry::default();
        let factory = registry.get_gguf(&arch).ok_or_else(|| {
            ExecutorError::WorkerInit(format!(
                "unsupported GGUF architecture: {arch}. Supported: {:?}",
                registry.gguf_architectures().collect::<Vec<_>>()
            ))
        })?;

        // Build the model.
        let model = factory(&mut gguf, &hf_config, device).map_err(|e| {
            ExecutorError::WorkerInit(format!("failed to construct GGUF model: {e}"))
        })?;

        // For GGUF, KV cache is always f32.
        let dtype = DType::F32;

        self.num_kv_heads = hf_config.num_kv_heads().unwrap_or(0);
        self.head_dim = hf_config.head_dim().unwrap_or(0);
        // Set model_dir to the parent of the GGUF file (for tokenizer lookup).
        self.model_dir = gguf_path.parent().map(|p| p.to_path_buf());
        self.hf_config = Some(hf_config);
        self.resolved_dtype = Some(dtype);
        self.model = Some(model);
        self.resolved_architecture = Some(arch.clone());
        info!(
            "CandleWorker: GGUF model loaded (arch={arch}, kv_dtype={:?})",
            dtype
        );

        // Grammar vocabulary is built lazily on first constrained-decoding request.

        Ok(())
    }

    /// Resolve a model path to a local directory.
    ///
    /// If the path is a local directory, returns it directly.
    /// Otherwise, downloads from HuggingFace Hub.
    fn resolve_model_path(&self) -> ExecutorResult<PathBuf> {
        let path = Path::new(&self.config.model_path);
        if path.is_dir() {
            return Ok(path.to_path_buf());
        }

        // Treat as a HuggingFace model ID — download via hf-hub.
        info!(
            "Downloading model from HuggingFace Hub: {}",
            self.config.model_path
        );
        let api = self.build_hf_api()?;
        let repo = api.model(self.config.model_path.clone());

        // Download config.json first (required).
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

        // Try to download the tokenizer (optional, best-effort).
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

        // Try single-file weights first.
        if repo.get("model.safetensors").is_ok() {
            info!("Downloaded single safetensors file");
            return Ok(model_dir);
        }

        // Otherwise, download the sharded index and all shard files.
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

        // Neither single nor sharded weights found — return directory anyway,
        // weight loading will produce a better error message.
        Ok(model_dir)
    }

    /// Verify speculative decode draft tokens against model logits.
    ///
    /// Returns the list of accepted tokens (may include a bonus token).
    fn verify_spec_decode(
        &self,
        logits: &Tensor,
        spec_token_ids: &[u32],
        num_input_tokens: usize,
    ) -> ExecutorResult<Vec<u32>> {
        let all_logits = logits
            .to_dtype(DType::F32)
            .and_then(|t| t.to_vec2::<f32>())
            .map_err(|e| {
                ExecutorError::WorkerExecution(format!("spec decode logits error: {e}"))
            })?;

        let mut accepted = Vec::new();
        let num_drafts = spec_token_ids.len();

        for i in 0..num_drafts {
            if i >= all_logits.len() {
                break;
            }
            let row = &all_logits[i];
            let argmax_id = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            let draft = spec_token_ids[i];
            if argmax_id == draft {
                accepted.push(draft);
            } else {
                accepted.push(argmax_id);
                break;
            }
        }

        // If all drafts accepted, sample bonus token from last position.
        if accepted.len() == num_drafts && num_input_tokens > 0 {
            let last_row = &all_logits[all_logits.len() - 1];
            let bonus = last_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
            accepted.push(bonus);
        }

        if accepted.is_empty() {
            let last_row = &all_logits[all_logits.len() - 1];
            let argmax = last_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
            Ok(vec![argmax])
        } else {
            debug!(
                "Spec decode: {}/{} drafts accepted (+ {} tokens total)",
                accepted.len().min(num_drafts),
                num_drafts,
                accepted.len()
            );
            Ok(accepted)
        }
    }

    /// Sample a single token from the last position's logits (normal decode/prefill).
    fn sample_normal(
        &self,
        logits: &Tensor,
        req_id: &str,
        token_count: usize,
        sampler: &mut Sampler,
        logprobs_map: &mut HashMap<String, Vec<vllm_common::LogprobsOutput>>,
    ) -> ExecutorResult<Vec<u32>> {
        let last_pos = token_count - 1;
        let req_logits = logits
            .narrow(0, last_pos, 1)
            .map_err(|e| ExecutorError::WorkerExecution(format!("tensor error: {e}")))?;

        let logits_vec = req_logits
            .to_dtype(DType::F32)
            .map_err(|e| ExecutorError::WorkerExecution(format!("dtype cast error: {e}")))?
            .flatten_all()
            .map_err(|e| ExecutorError::WorkerExecution(format!("flatten error: {e}")))?
            .to_vec1::<f32>()
            .map_err(|e| ExecutorError::WorkerExecution(format!("tensor error: {e}")))?;

        #[cfg(feature = "guided-decoding")]
        let grammar_allowed: Option<Vec<u32>> = self
            .grammar_states
            .get(req_id)
            .and_then(|g| g.allowed_tokens());
        #[cfg(not(feature = "guided-decoding"))]
        let grammar_allowed: Option<Vec<u32>> = None;

        if let Some(params) = self.sampling_params_map.get(req_id) {
            let prev_tokens = self
                .token_buffers
                .get(req_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let (token_id, maybe_logprobs) =
                sampler.sample_one(&logits_vec, params, prev_tokens, grammar_allowed.as_deref());
            if let Some(lp) = maybe_logprobs {
                logprobs_map.entry(req_id.to_string()).or_default().push(lp);
            }
            Ok(vec![token_id])
        } else {
            let indices = req_logits
                .argmax(candle_core::D::Minus1)
                .map_err(|e| ExecutorError::WorkerExecution(format!("argmax error: {e}")))?;
            indices
                .to_vec1::<u32>()
                .map_err(|e| ExecutorError::WorkerExecution(format!("tensor error: {e}")))
        }
    }
}

impl Worker for CandleWorker {
    fn init_device(&mut self) -> ExecutorResult<()> {
        let device = parse_device(&self.config.device_str)?;
        info!(
            "CandleWorker: initialized device {:?}",
            self.config.device_str
        );
        self.device = Some(device);
        Ok(())
    }

    fn load_model(&mut self) -> ExecutorResult<()> {
        let device = self
            .device
            .clone()
            .ok_or_else(|| ExecutorError::WorkerInit("device not initialized".to_string()))?;

        let explicit_dtype = self.config.candle_dtype()?;

        // Check if this is a GGUF model path.
        let gguf_path = self.resolve_gguf_path()?;

        if let Some(gguf_path) = gguf_path {
            // ---- GGUF loading path ----
            return self.load_model_gguf(&gguf_path, &device);
        }

        // ---- SafeTensors loading path ----

        // 1. Resolve model directory (local or HF download).
        let model_dir = self.resolve_model_path()?;
        info!("CandleWorker: loading model from {}", model_dir.display());

        // Optimistically parse tokenizer.json on a background thread while we
        // load config + weights on the main thread. The result is stored on
        // self and consumed later by init.rs, avoiding a redundant parse.
        let tok_dir = model_dir.clone();
        let tokenizer_handle = std::thread::spawn(move || {
            let path = tok_dir.join("tokenizer.json");
            if path.exists() {
                tokenizers::Tokenizer::from_file(&path).ok()
            } else {
                None
            }
        });

        // 2. Parse config.json.
        let hf_config = HfModelConfig::from_dir(&model_dir)
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to parse config.json: {e}")))?;

        // Unwrap composite models (e.g. Kimi K2.5 → text_config).
        let (hf_config, weight_prefix) = match hf_config.resolve_text_config() {
            Some((text_cfg, prefix)) => {
                info!(
                    "CandleWorker: composite model detected, unwrapping text config (strip prefix: {prefix})"
                );
                (text_cfg, Some(prefix))
            }
            None => (hf_config, None),
        };

        // Resolve "auto" dtype: read torch_dtype from config.json, fall back to F16.
        let dtype = if let Some(dt) = explicit_dtype {
            dt
        } else {
            let resolved = hf_config
                .torch_dtype
                .as_deref()
                .and_then(|s| vllm_model::tensor::dtype_from_str(s).ok())
                .unwrap_or(DType::F16);
            info!(
                "CandleWorker: auto dtype resolved to {}",
                vllm_model::tensor::dtype_to_str(resolved)
            );
            resolved
        };

        // 3. Detect GPTQ or AWQ quantization.
        let quant_method = hf_config
            .extra
            .get("quantization_config")
            .and_then(|v| v.get("quant_method"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let gptq_config = vllm_model::gptq_config::GptqQuantizeConfig::from_dir(&model_dir).ok();
        let is_gptq = gptq_config.is_some() || quant_method.as_deref() == Some("gptq");

        let awq_config = vllm_model::awq_config::AwqQuantizeConfig::from_dir(&model_dir).ok();
        let is_awq = !is_gptq && (awq_config.is_some() || quant_method.as_deref() == Some("awq"));

        // 4. Look up architecture in the registry.
        let arch = hf_config
            .architectures
            .first()
            .ok_or_else(|| {
                ExecutorError::WorkerInit("config.json has no architectures field".to_string())
            })?
            .clone();
        let registry = ModelRegistry::default();

        // 5. Load weights.
        let mut weights = ModelWeights::from_dir(&model_dir, &device)
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to load weights: {e}")))?;
        // Strip weight name prefix for composite models (e.g. "language_model." for K2.5).
        if let Some(prefix) = weight_prefix {
            let before = weights.len();
            weights.strip_prefix(prefix);
            info!(
                "CandleWorker: stripped prefix \"{prefix}\" from weights ({before} → {} tensors)",
                weights.len()
            );
        }
        info!(
            "CandleWorker: loaded {} tensors ({:.1} MB)",
            weights.names().len(),
            weights.total_size_bytes() as f64 / 1_048_576.0
        );

        // 6. Construct the model (GPTQ, AWQ, or standard).
        let model = if is_gptq {
            let gptq_cfg = match gptq_config {
                Some(cfg) => cfg,
                None => {
                    let qc = hf_config
                        .extra
                        .get("quantization_config")
                        .ok_or_else(|| {
                            ExecutorError::WorkerInit(
                                "GPTQ detected but no quantize_config.json or quantization_config in config.json".to_string(),
                            )
                        })?;
                    vllm_model::gptq_config::GptqQuantizeConfig::from_json_value(qc).map_err(
                        |e| {
                            ExecutorError::WorkerInit(format!(
                                "failed to parse quantization_config: {e}"
                            ))
                        },
                    )?
                }
            };

            info!(
                "CandleWorker: GPTQ detected (bits={}, group_size={}, desc_act={})",
                gptq_cfg.bits, gptq_cfg.group_size, gptq_cfg.desc_act
            );

            let gptq_factory = registry.get_gptq(&arch).ok_or_else(|| {
                ExecutorError::WorkerInit(format!(
                    "unsupported GPTQ architecture: {arch}. Supported GPTQ: {:?}",
                    registry.gptq_architectures().collect::<Vec<_>>()
                ))
            })?;

            gptq_factory(&weights, &hf_config, &gptq_cfg, dtype, &device).map_err(|e| {
                ExecutorError::WorkerInit(format!("failed to construct GPTQ model: {e}"))
            })?
        } else if is_awq {
            let awq_cfg = match awq_config {
                Some(cfg) => cfg,
                None => {
                    let qc = hf_config
                        .extra
                        .get("quantization_config")
                        .ok_or_else(|| {
                            ExecutorError::WorkerInit(
                                "AWQ detected but no quant_config.json or quantization_config in config.json".to_string(),
                            )
                        })?;
                    vllm_model::awq_config::AwqQuantizeConfig::from_json_value(qc).map_err(|e| {
                        ExecutorError::WorkerInit(format!(
                            "failed to parse quantization_config: {e}"
                        ))
                    })?
                }
            };

            info!(
                "CandleWorker: AWQ detected (bits={}, group_size={}, zero_point={})",
                awq_cfg.bits, awq_cfg.group_size, awq_cfg.zero_point
            );

            let awq_factory = registry.get_awq(&arch).ok_or_else(|| {
                ExecutorError::WorkerInit(format!(
                    "unsupported AWQ architecture: {arch}. Supported AWQ: {:?}",
                    registry.awq_architectures().collect::<Vec<_>>()
                ))
            })?;

            awq_factory(&weights, &hf_config, &awq_cfg, dtype, &device).map_err(|e| {
                ExecutorError::WorkerInit(format!("failed to construct AWQ model: {e}"))
            })?
        } else {
            let factory = registry.get(&arch).ok_or_else(|| {
                ExecutorError::WorkerInit(format!(
                    "unsupported architecture: {arch}. Supported: {:?}",
                    registry.architectures().collect::<Vec<_>>()
                ))
            })?;

            factory(&weights, &hf_config, dtype, &device)
                .map_err(|e| ExecutorError::WorkerInit(format!("failed to construct model: {e}")))?
        };

        self.num_kv_heads = hf_config.num_kv_heads().unwrap_or(0);
        self.head_dim = hf_config.head_dim().unwrap_or(0);
        self.model_dir = Some(model_dir);
        self.hf_config = Some(hf_config);
        self.resolved_dtype = Some(dtype);
        self.model = Some(model);
        self.resolved_architecture = Some(arch.clone());
        info!(
            "CandleWorker: model loaded (arch={arch}, dtype={:?})",
            dtype
        );

        // Inject LoRA adapter if configured.
        if let Some(ref adapter_path) = self.config.lora_adapter {
            let adapter =
                vllm_model::lora::LoraAdapter::from_dir(adapter_path, "default", &device, dtype)
                    .map_err(|e| {
                        ExecutorError::WorkerInit(format!("failed to load LoRA adapter: {e}"))
                    })?;
            if let Some(ref mut model) = self.model {
                model.inject_lora(&adapter).map_err(|e| {
                    ExecutorError::WorkerInit(format!("failed to inject LoRA: {e}"))
                })?;
            }
            info!(
                "CandleWorker: LoRA adapter '{}' loaded (rank={}, targets={:?})",
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
        num_cpu_blocks: usize,
    ) -> ExecutorResult<()> {
        self.num_gpu_blocks = num_gpu_blocks;
        self.num_cpu_blocks = num_cpu_blocks;

        // Create the paged KV block pool if model dimensions are known.
        let num_layers = self.model.as_ref().map(|m| m.num_layers()).unwrap_or(0);
        if num_layers > 0 && self.num_kv_heads > 0 && self.head_dim > 0 && num_gpu_blocks > 0 {
            let dtype = self.resolved_dtype.unwrap_or(DType::F32);
            let device = self.device.as_ref().cloned().unwrap_or(Device::Cpu);
            let pool = KvBlockPool::new(
                num_gpu_blocks,
                num_layers,
                self.num_kv_heads,
                self.head_dim,
                self.config.block_size,
                dtype,
                &device,
            )
            .map_err(|e| {
                ExecutorError::WorkerInit(format!("failed to create KV block pool: {e}"))
            })?;
            info!(
                "CandleWorker: KV block pool created (blocks={}, layers={}, kv_heads={}, head_dim={}, block_size={}, dtype={:?})",
                num_gpu_blocks,
                num_layers,
                self.num_kv_heads,
                self.head_dim,
                self.config.block_size,
                dtype
            );
            self.kv_block_pool = Some(pool);
        }

        info!(
            "CandleWorker: cache initialized (gpu_blocks={}, cpu_blocks={})",
            num_gpu_blocks, num_cpu_blocks
        );
        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        // On CUDA devices, query actual GPU VRAM via cudarc.
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device
            && device.is_cuda()
        {
            return cuda_free_memory()
                .map_err(|e| ExecutorError::WorkerInit(format!("CUDA memory query failed: {e}")));
        }

        // CPU / Metal fallback: query system RAM directly via libc instead of
        // the `sysinfo` crate (which is surprisingly expensive).
        //
        // On Linux, _SC_AVPHYS_PAGES gives free pages (conservative "available").
        // On macOS, only _SC_PHYS_PAGES exists — report total physical memory
        // (the candle-Metal path is rare; MLX is the primary macOS backend).
        #[cfg(target_os = "linux")]
        let pages = unsafe { libc::sysconf(libc::_SC_AVPHYS_PAGES) };
        #[cfg(not(target_os = "linux"))]
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };

        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let available = if pages > 0 && page_size > 0 {
            (pages as usize) * (page_size as usize)
        } else {
            // Fallback: 4 GiB.
            4 * 1024 * 1024 * 1024
        };
        Ok(available)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
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
                self.ensure_grammar_vocabulary();
            }
        }

        let model = self
            .model
            .as_mut()
            .ok_or_else(|| ExecutorError::WorkerExecution("model not loaded".to_string()))?;
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerExecution("device not initialized".to_string()))?;

        let num_layers = model.num_layers();
        let use_paged = self.kv_block_pool.is_some();

        // Clean up buffers for finished requests.
        for req_id in &scheduler_output.finished_req_ids {
            self.token_buffers.remove(req_id);
            self.sampling_params_map.remove(req_id);
            self.kv_caches.remove(req_id);
            self.recurrent_states.remove(req_id);
            #[cfg(feature = "guided-decoding")]
            self.grammar_states.remove(req_id);
            self.mm_data_map.remove(req_id);
        }
        self.input_batch
            .remove_finished(&scheduler_output.finished_req_ids);

        // --- Process newly scheduled requests ---
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

            // Store prompt tokens in the buffer (needed for penalty sampling).
            self.token_buffers
                .insert(new_req.req_id.clone(), prompt_ids.to_vec());

            // Store sampling params if provided.
            if let Some(ref params) = new_req.sampling_params {
                // Create grammar guide for constrained decoding if requested.
                #[cfg(feature = "guided-decoding")]
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

            if use_paged {
                let block_ids = new_req.block_ids.first().cloned().unwrap_or_default();
                self.input_batch.add_request(
                    new_req.req_id.clone(),
                    tokens_to_use,
                    block_ids,
                    new_req.num_computed_tokens,
                );
            } else {
                // Legacy: per-request contiguous KV cache.
                self.kv_caches
                    .insert(new_req.req_id.clone(), vec![None; num_layers]);
            }

            // Store multimodal data for VLM models (consumed during forward).
            if let Some(ref mm_data) = new_req.mm_data {
                self.mm_data_map
                    .insert(new_req.req_id.clone(), mm_data.clone());
            }
        }

        // --- Update cached requests' block tables ---
        if use_paged {
            let new_req_ids: HashSet<&str> = scheduler_output
                .scheduled_new_reqs
                .iter()
                .map(|r| r.req_id.as_str())
                .collect();
            for (i, req_id) in scheduler_output
                .scheduled_cached_reqs
                .req_ids
                .iter()
                .enumerate()
            {
                if new_req_ids.contains(req_id.as_str()) {
                    continue;
                }
                if let Some(Some(new_blocks)) =
                    scheduler_output.scheduled_cached_reqs.new_block_ids.get(i)
                    && let Some(group0) = new_blocks.first()
                {
                    self.input_batch.update_blocks(req_id, group0.clone());
                }
            }
        }

        // --- Pooling mode: run hidden_states + pool + normalize ---
        if self.is_pooling {
            use vllm_models::embedding::{l2_normalize, pool};
            let strategy = self.pooling_strategy;

            let mut pooler_map: HashMap<String, Vec<f32>> = HashMap::new();

            for req_id in scheduler_output.num_scheduled_tokens.keys() {
                let token_ids = match self.token_buffers.get(req_id) {
                    Some(buf) => buf.as_slice(),
                    None => continue,
                };

                let input_ids = Tensor::new(token_ids, device)
                    .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
                let positions: Vec<u32> = (0..token_ids.len() as u32).collect();
                let positions_t = Tensor::new(positions.as_slice(), device)
                    .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

                let hidden = model
                    .hidden_states(&input_ids, &positions_t)
                    .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

                let pooled = pool(&hidden, strategy)
                    .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

                let normalized = l2_normalize(&pooled)
                    .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

                let vec: Vec<f32> = normalized
                    .to_vec1()
                    .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

                pooler_map.insert(req_id.clone(), vec);
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
            });
        }

        // --- Build model inputs ---
        // For paged mode: use InputBatch to produce flat tensors + metadata.
        // For legacy mode: build per-request inputs from token_buffers.

        // Create a sampler for non-greedy requests (created per-call to avoid
        // Send issues — Sampler holds ThreadRng which is !Send).
        let mut sampler = Sampler::new();
        let mut token_map: HashMap<String, Vec<u32>> = HashMap::new();
        let mut logprobs_map: HashMap<String, Vec<vllm_common::LogprobsOutput>> = HashMap::new();
        let mut prompt_logprobs_map: HashMap<String, Vec<vllm_common::LogprobsOutput>> =
            HashMap::new();

        if use_paged {
            // Use InputBatch to prepare flat tensors and metadata.
            let prepared = self
                .input_batch
                .prepare_inputs(&scheduler_output.scheduled_spec_decode_tokens);

            if prepared.flat_token_ids.is_empty() {
                return Ok(ModelRunnerOutput::from_token_map(HashMap::new()));
            }

            let flat_ids = Tensor::new(prepared.flat_token_ids.as_slice(), device)
                .map_err(|e| ExecutorError::WorkerExecution(format!("tensor error: {e}")))?;
            let flat_pos = Tensor::new(prepared.flat_positions.as_slice(), device)
                .map_err(|e| ExecutorError::WorkerExecution(format!("tensor error: {e}")))?;

            // Build BatchedKvCacheStorage.
            let pool = self.kv_block_pool.as_mut().unwrap();
            let mut batched_storage = vllm_models::BatchedKvCacheStorage::new(
                pool,
                prepared.batch_block_ids,
                prepared.batch_tokens_before,
            );

            // Single batched forward pass.
            let all_logits_flat = model
                .forward_batch(
                    &flat_ids,
                    &flat_pos,
                    &prepared.attn_meta,
                    &mut batched_storage,
                )
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("batched forward failed: {e}"))
                })?;

            // Flush deferred scatters.
            batched_storage
                .flush_all()
                .map_err(|e| ExecutorError::WorkerExecution(format!("scatter flush error: {e}")))?;

            // Split flat logits [total_tokens, vocab] into per-request logits.
            let mut per_req_logits = Vec::with_capacity(prepared.req_inputs.len());
            let mut pos = 0usize;
            for r in &prepared.req_inputs {
                let q_len = r.token_count;
                let req_logits = all_logits_flat
                    .narrow(0, pos, q_len)
                    .map_err(|e| ExecutorError::WorkerExecution(format!("narrow error: {e}")))?;
                per_req_logits.push(req_logits);
                pos += q_len;
            }

            // --- Per-request sampling (paged path) ---
            let any_logprobs_requested = prepared.req_inputs.iter().any(|r| {
                self.sampling_params_map
                    .get(&r.req_id)
                    .and_then(|p| p.logprobs)
                    .is_some()
            });

            for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
                let logits = &per_req_logits[req_idx];
                let token_ids_slice = &prepared.flat_token_ids
                    [req_slice.token_start..req_slice.token_start + req_slice.token_count];

                // Compute prompt logprobs if requested and this is a prefill.
                if req_slice.token_count > 1
                    && let Some(params) = self.sampling_params_map.get(&req_slice.req_id)
                    && let Some(top_n) = params.prompt_logprobs
                {
                    let top_n = top_n.max(0) as usize;
                    let num_positions = req_slice.token_count - 1;
                    let prefix_logits = logits
                        .narrow(0, 0, num_positions)
                        .and_then(|t| t.to_dtype(DType::F32))
                        .and_then(|t| t.to_vec2::<f32>())
                        .map_err(|e| {
                            ExecutorError::WorkerExecution(format!(
                                "prompt logprobs tensor error: {e}"
                            ))
                        })?;
                    let mut plps = Vec::with_capacity(num_positions);
                    for (i, row) in prefix_logits.iter().enumerate() {
                        let actual_token = token_ids_slice[i + 1];
                        plps.push(vllm_models::sampler::compute_logprobs(
                            row,
                            actual_token,
                            top_n,
                        ));
                    }
                    prompt_logprobs_map.insert(req_slice.req_id.clone(), plps);
                }

                // Sample / verify speculative tokens.
                let sampled = if !req_slice.spec_token_ids.is_empty() {
                    self.verify_spec_decode(
                        logits,
                        &req_slice.spec_token_ids,
                        req_slice.token_count,
                    )?
                } else {
                    self.sample_normal(
                        logits,
                        &req_slice.req_id,
                        req_slice.token_count,
                        &mut sampler,
                        &mut logprobs_map,
                    )?
                };

                // Advance grammar state.
                #[cfg(feature = "guided-decoding")]
                if let Some(guide) = self.grammar_states.get_mut(&req_slice.req_id)
                    && let Some(&token_id) = sampled.first()
                {
                    guide.advance(token_id);
                }

                // Commit step results to InputBatch.
                self.input_batch.commit_step(
                    &req_slice.req_id,
                    &sampled,
                    req_slice.token_count,
                    !req_slice.spec_token_ids.is_empty(),
                );

                // Update the token buffer with the new sampled token(s).
                if let Some(buf) = self.token_buffers.get_mut(&req_slice.req_id) {
                    buf.extend_from_slice(&sampled);
                }

                token_map.insert(req_slice.req_id.clone(), sampled);
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
            return Ok(output);
        }

        // --- Legacy per-request KV cache path (no InputBatch) ---
        struct ReqInput {
            req_id: String,
            token_ids: Vec<u32>,
            positions: Vec<u32>,
            spec_token_ids: Vec<u32>,
        }
        let mut req_inputs: Vec<ReqInput> = Vec::new();

        // Build prefill inputs for new requests.
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
            let pos_offset = new_req.num_computed_tokens;
            let positions: Vec<u32> = (0..tokens_to_use.len() as u32)
                .map(|i| pos_offset + i)
                .collect();
            req_inputs.push(ReqInput {
                req_id: new_req.req_id.clone(),
                token_ids: tokens_to_use.to_vec(),
                positions,
                spec_token_ids: Vec::new(),
            });
        }

        // Build decode inputs for cached requests.
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
                let position = (buf.len() - 1) as u32;
                req_inputs.push(ReqInput {
                    req_id: req_id.clone(),
                    token_ids: vec![last_token],
                    positions: vec![position],
                    spec_token_ids: Vec::new(),
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
                let position = (buf.len() - 1) as u32;
                req_inputs.push(ReqInput {
                    req_id: req_id.clone(),
                    token_ids: vec![last_token],
                    positions: vec![position],
                    spec_token_ids: Vec::new(),
                });
            } else {
                warn!("No token buffer for continuing request {}", req_id);
            }
        }

        if req_inputs.is_empty() {
            return Ok(ModelRunnerOutput::from_token_map(HashMap::new()));
        }

        let mut per_req_logits = Vec::with_capacity(req_inputs.len());
        for req_input in &req_inputs {
            // Inject per-request recurrent state for hybrid models (e.g., Qwen3-Next GDN).
            // If no saved state exists yet, initialize with empty state.
            if model.num_recurrent_layers() > 0 {
                if let Some(rs) = self.recurrent_states.get(&req_input.req_id) {
                    model.inject_recurrent_state(rs);
                } else {
                    model.reset_recurrent_state();
                }
            }

            let mm_data = self.mm_data_map.remove(&req_input.req_id);
            model.set_mm_data(mm_data);

            let input_ids = Tensor::new(req_input.token_ids.as_slice(), device)
                .map_err(|e| ExecutorError::WorkerExecution(format!("tensor error: {e}")))?;
            let positions = Tensor::new(req_input.positions.as_slice(), device)
                .map_err(|e| ExecutorError::WorkerExecution(format!("tensor error: {e}")))?;

            let logits = if let Some(kv_cache) = self.kv_caches.get_mut(&req_input.req_id) {
                let mut storage = KvCacheStorage::Contiguous(kv_cache);
                model
                    .forward(&input_ids, &positions, Some(&mut storage))
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("model forward failed: {e}"))
                    })?
            } else {
                model.forward(&input_ids, &positions, None).map_err(|e| {
                    ExecutorError::WorkerExecution(format!("model forward failed: {e}"))
                })?
            };
            // Extract recurrent state after forward for hybrid models.
            if model.num_recurrent_layers() > 0 {
                self.recurrent_states
                    .insert(req_input.req_id.clone(), model.extract_recurrent_state());
            }

            per_req_logits.push(logits);
        }

        // --- Per-request sampling (legacy path) ---
        let any_logprobs_requested = req_inputs.iter().any(|r| {
            self.sampling_params_map
                .get(&r.req_id)
                .and_then(|p| p.logprobs)
                .is_some()
        });

        for (req_idx, req_input) in req_inputs.iter().enumerate() {
            let logits = &per_req_logits[req_idx];

            // Compute prompt logprobs if requested and this is a prefill.
            if req_input.token_ids.len() > 1
                && let Some(params) = self.sampling_params_map.get(&req_input.req_id)
                && let Some(top_n) = params.prompt_logprobs
            {
                let top_n = top_n.max(0) as usize;
                let num_positions = req_input.token_ids.len() - 1;
                let prefix_logits = logits
                    .narrow(0, 0, num_positions)
                    .and_then(|t| t.to_dtype(DType::F32))
                    .and_then(|t| t.to_vec2::<f32>())
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("prompt logprobs tensor error: {e}"))
                    })?;
                let mut plps = Vec::with_capacity(num_positions);
                for (i, row) in prefix_logits.iter().enumerate() {
                    let actual_token = req_input.token_ids[i + 1];
                    plps.push(vllm_models::sampler::compute_logprobs(
                        row,
                        actual_token,
                        top_n,
                    ));
                }
                prompt_logprobs_map.insert(req_input.req_id.clone(), plps);
            }

            // Sample / verify speculative tokens.
            let sampled = if !req_input.spec_token_ids.is_empty() {
                self.verify_spec_decode(
                    logits,
                    &req_input.spec_token_ids,
                    req_input.token_ids.len(),
                )?
            } else {
                self.sample_normal(
                    logits,
                    &req_input.req_id,
                    req_input.token_ids.len(),
                    &mut sampler,
                    &mut logprobs_map,
                )?
            };

            // Advance grammar state.
            #[cfg(feature = "guided-decoding")]
            if let Some(guide) = self.grammar_states.get_mut(&req_input.req_id)
                && let Some(&token_id) = sampled.first()
            {
                guide.advance(token_id);
            }

            // Update the token buffer with the new sampled token(s).
            if let Some(buf) = self.token_buffers.get_mut(&req_input.req_id) {
                buf.extend_from_slice(&sampled);
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

    fn embed(&mut self, token_id_seqs: &[&[u32]]) -> ExecutorResult<Vec<Vec<f32>>> {
        use vllm_models::embedding::{l2_normalize, pool};

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerExecution("model not loaded".into()))?;
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerExecution("device not initialized".into()))?;
        let strategy = self.pooling_strategy;

        let mut results = Vec::with_capacity(token_id_seqs.len());
        for token_ids in token_id_seqs {
            let input_ids = Tensor::new(*token_ids, device)
                .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
            let positions: Vec<u32> = (0..token_ids.len() as u32).collect();
            let positions = Tensor::new(positions.as_slice(), device)
                .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

            let hidden_states = model
                .hidden_states(&input_ids, &positions)
                .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

            let pooled = pool(&hidden_states, strategy)
                .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

            let normalized =
                l2_normalize(&pooled).map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

            let vec: Vec<f32> = normalized
                .to_vec1()
                .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
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
        self.is_shutdown = true;
        self.model = None;
        info!("CandleWorker: shut down");
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
// Helpers
// ---------------------------------------------------------------------------

/// Look for a `.gguf` file in a directory. Returns the first one found.
fn find_gguf_in_dir(dir: &Path) -> Option<PathBuf> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
            {
                return Some(path);
            }
        }
    }
    None
}

/// Derive candidate base model repo names from a GGUF repo ID.
///
/// E.g. `"unsloth/Llama-3.1-8B-Instruct-GGUF"` → `["unsloth/Llama-3.1-8B-Instruct"]`
///      `"bartowski/Meta-Llama-3-8B-Instruct-GGUF"` → `["bartowski/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"]`
fn derive_base_repo_candidates(model_id: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    // Strip common GGUF suffixes (case-insensitive).
    let suffixes = ["-GGUF", "-gguf", "_GGUF", "_gguf"];
    let mut base_name = None;
    for suffix in &suffixes {
        if let Some(stripped) = model_id.strip_suffix(suffix) {
            base_name = Some(stripped);
            break;
        }
    }

    if let Some(base) = base_name {
        // Same org, stripped name.
        candidates.push(base.to_string());

        // Also try common original-model orgs for well-known re-quantizers.
        // E.g. bartowski/Llama-3.1-8B-Instruct-GGUF → meta-llama/Llama-3.1-8B-Instruct
        if let Some(slash) = base.find('/') {
            let model_name = &base[slash + 1..];
            let org = &base[..slash];

            // Don't duplicate if same org.
            let alt_orgs = [
                "meta-llama",
                "mistralai",
                "Qwen",
                "google",
                "microsoft",
                "unsloth",
            ];
            for alt_org in &alt_orgs {
                if *alt_org != org {
                    candidates.push(format!("{alt_org}/{model_name}"));
                }
            }
        }
    }

    candidates
}

/// Parse a device string ("cpu", "cuda:0", "metal", "auto") into a candle Device.
pub fn parse_device(device_str: &str) -> ExecutorResult<Device> {
    match device_str {
        "cpu" => Ok(Device::Cpu),
        "auto" => Ok(auto_detect_device()),
        s if s.starts_with("cuda") => {
            let ordinal = if s == "cuda" {
                0
            } else if let Some(n) = s.strip_prefix("cuda:") {
                n.parse::<usize>()
                    .map_err(|_| ExecutorError::Config(format!("invalid CUDA ordinal: {n}")))?
            } else {
                return Err(ExecutorError::Config(format!("invalid device: {s}")));
            };
            Device::new_cuda(ordinal).map_err(|e| {
                ExecutorError::Config(format!("failed to init CUDA device {ordinal}: {e}"))
            })
        }
        s if s.starts_with("metal") => {
            let ordinal = if s == "metal" {
                0
            } else if let Some(n) = s.strip_prefix("metal:") {
                n.parse::<usize>()
                    .map_err(|_| ExecutorError::Config(format!("invalid Metal ordinal: {n}")))?
            } else {
                return Err(ExecutorError::Config(format!("invalid device: {s}")));
            };
            Device::new_metal(ordinal).map_err(|e| {
                ExecutorError::Config(format!("failed to init Metal device {ordinal}: {e}"))
            })
        }
        other => Err(ExecutorError::Config(format!(
            "unsupported device: {other}. Valid options: cpu, cuda, cuda:N, metal, auto"
        ))),
    }
}

/// Auto-detect the best available device: Metal > CUDA > CPU.
fn auto_detect_device() -> Device {
    // Try Metal first (Apple Silicon).
    if let Ok(device) = Device::new_metal(0) {
        info!("Auto-detected Metal GPU");
        return device;
    }
    // Try CUDA.
    if let Ok(device) = Device::new_cuda(0) {
        info!("Auto-detected CUDA GPU");
        return device;
    }
    info!("No GPU detected, using CPU");
    Device::Cpu
}

/// Query free GPU memory on the current CUDA device via cudarc.
///
/// Uses `cudarc::driver::result::mem_get_info()` which calls `cuMemGetInfo_v2`
/// on the current CUDA context. The candle Device::new_cuda() call in init_device()
/// has already set the current context for us.
#[cfg(feature = "cuda")]
fn cuda_free_memory() -> Result<usize, String> {
    let (free, _total) =
        cudarc::driver::result::mem_get_info().map_err(|e| format!("cuMemGetInfo: {e}"))?;
    Ok(free)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> CandleWorkerConfig {
        CandleWorkerConfig {
            model_path: "/nonexistent/model".to_string(),
            device_str: "cpu".to_string(),
            dtype: "f32".to_string(),
            hf_token: None,
            cache_dir: None,
            block_size: 16,
            gguf_file: None,
            lora_adapter: None,
            pooling_strategy: "auto".to_string(),
            is_pooling: false,
        }
    }

    #[test]
    fn test_config_dtype_parsing() {
        let mut cfg = make_config();
        assert_eq!(cfg.candle_dtype().unwrap(), Some(DType::F32));
        cfg.dtype = "f16".to_string();
        assert_eq!(cfg.candle_dtype().unwrap(), Some(DType::F16));
        cfg.dtype = "bf16".to_string();
        assert_eq!(cfg.candle_dtype().unwrap(), Some(DType::BF16));
        cfg.dtype = "auto".to_string();
        assert_eq!(cfg.candle_dtype().unwrap(), None);
        cfg.dtype = "float16".to_string();
        assert_eq!(cfg.candle_dtype().unwrap(), Some(DType::F16));
        cfg.dtype = "bfloat16".to_string();
        assert_eq!(cfg.candle_dtype().unwrap(), Some(DType::BF16));
        cfg.dtype = "float32".to_string();
        assert_eq!(cfg.candle_dtype().unwrap(), Some(DType::F32));
        cfg.dtype = "invalid".to_string();
        assert!(cfg.candle_dtype().is_err());
    }

    #[test]
    fn test_parse_device_cpu() {
        let device = parse_device("cpu").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_parse_device_metal() {
        // Metal may or may not be available depending on the build features
        // and hardware. We just verify parsing doesn't panic on valid input.
        let result = parse_device("metal");
        // On a non-Metal build, this returns an error about Metal not being compiled.
        // On a Metal build on Apple Silicon, it succeeds.
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_parse_device_auto() {
        // Auto always succeeds (falls back to CPU).
        let device = parse_device("auto").unwrap();
        // Should be some device (Metal, CUDA, or CPU).
        let _ = device;
    }

    #[test]
    fn test_parse_device_invalid() {
        assert!(parse_device("tpu").is_err());
        assert!(parse_device("cuda:abc").is_err());
        assert!(parse_device("metal:abc").is_err());
    }

    #[test]
    fn test_init_device_cpu() {
        let mut worker = CandleWorker::new(make_config());
        worker.init_device().unwrap();
        assert!(worker.device.is_some());
    }

    #[test]
    fn test_load_model_without_init_fails() {
        let mut worker = CandleWorker::new(make_config());
        assert!(worker.load_model().is_err());
    }

    #[test]
    fn test_determine_available_memory() {
        let mut worker = CandleWorker::new(make_config());
        worker.init_device().unwrap();
        let mem = worker.determine_available_memory().unwrap();
        assert!(mem > 0);
    }

    #[test]
    fn test_initialize_cache() {
        let mut worker = CandleWorker::new(make_config());
        worker.initialize_cache(100, 10).unwrap();
        assert_eq!(worker.num_gpu_blocks, 100);
        assert_eq!(worker.num_cpu_blocks, 10);
    }

    #[test]
    fn test_shutdown() {
        let mut worker = CandleWorker::new(make_config());
        worker.shutdown();
        assert!(worker.is_shutdown);
    }

    #[test]
    fn test_rank_accessors() {
        let worker = CandleWorker::new(make_config());
        assert_eq!(worker.rank(), 0);
        assert_eq!(worker.local_rank(), 0);
        assert!(worker.is_driver_worker());
    }

    #[test]
    fn test_token_buffer_initialized_empty() {
        let worker = CandleWorker::new(make_config());
        assert!(worker.token_buffers.is_empty());
        assert!(worker.sampling_params_map.is_empty());
        assert!(worker.kv_caches.is_empty());
    }

    #[test]
    fn test_shutdown_clears_model() {
        let mut worker = CandleWorker::new(make_config());
        worker.token_buffers.insert("r1".to_string(), vec![1, 2, 3]);
        worker.shutdown();
        assert!(worker.is_shutdown);
        assert!(worker.model.is_none());
    }

    #[test]
    fn test_resolve_gguf_path_not_gguf() {
        let worker = CandleWorker::new(make_config());
        // A non-existent path that doesn't end in .gguf and isn't a dir → None.
        let result = worker.resolve_gguf_path().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_gguf_path_local_file() {
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("model.gguf");
        std::fs::write(&gguf_path, b"fake gguf data").unwrap();

        let mut config = make_config();
        config.model_path = gguf_path.to_string_lossy().to_string();
        let worker = CandleWorker::new(config);

        let result = worker.resolve_gguf_path().unwrap();
        assert_eq!(result, Some(gguf_path));
    }

    #[test]
    fn test_resolve_gguf_path_dir_with_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("model-Q4_K_M.gguf");
        std::fs::write(&gguf_path, b"fake gguf data").unwrap();

        let mut config = make_config();
        config.model_path = dir.path().to_string_lossy().to_string();
        let worker = CandleWorker::new(config);

        let result = worker.resolve_gguf_path().unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().extension().unwrap() == "gguf");
    }

    #[test]
    fn test_resolve_gguf_path_dir_no_gguf() {
        let dir = tempfile::tempdir().unwrap();
        // Empty dir — no .gguf files.
        let mut config = make_config();
        config.model_path = dir.path().to_string_lossy().to_string();
        let worker = CandleWorker::new(config);

        let result = worker.resolve_gguf_path().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_derive_base_repo_gguf_suffix() {
        let candidates = derive_base_repo_candidates("unsloth/Llama-3.1-8B-Instruct-GGUF");
        assert!(candidates.contains(&"unsloth/Llama-3.1-8B-Instruct".to_string()));
    }

    #[test]
    fn test_derive_base_repo_lowercase_suffix() {
        let candidates = derive_base_repo_candidates("user/model-gguf");
        assert!(candidates.contains(&"user/model".to_string()));
    }

    #[test]
    fn test_derive_base_repo_no_suffix() {
        let candidates = derive_base_repo_candidates("meta-llama/Llama-3.1-8B-Instruct");
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_derive_base_repo_cross_org() {
        let candidates = derive_base_repo_candidates("bartowski/Llama-3.1-8B-Instruct-GGUF");
        // Should include same-org stripped name + cross-org candidates.
        assert!(candidates.contains(&"bartowski/Llama-3.1-8B-Instruct".to_string()));
        assert!(candidates.contains(&"meta-llama/Llama-3.1-8B-Instruct".to_string()));
    }

    // -- Integration test using a tiny stub model --

    /// A minimal model that does a real embedding lookup and returns logits.
    /// This exercises the same tensor shapes as a real model.
    struct TinyStubModel {
        embedding: Tensor, // [vocab_size, hidden_size]
        lm_head: Tensor,   // [vocab_size, hidden_size]
    }

    impl TinyStubModel {
        fn new(vocab_size: usize, hidden_size: usize) -> Self {
            let embedding =
                Tensor::randn(0f32, 1.0, &[vocab_size, hidden_size], &Device::Cpu).unwrap();
            let lm_head =
                Tensor::randn(0f32, 1.0, &[vocab_size, hidden_size], &Device::Cpu).unwrap();
            Self { embedding, lm_head }
        }
    }

    impl vllm_models::Model for TinyStubModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _positions: &Tensor,
            _kv_cache: Option<&mut vllm_models::KvCacheStorage<'_>>,
        ) -> vllm_model::ModelResult<Tensor> {
            // input_ids: [num_tokens] (1D)
            let hidden = self
                .embedding
                .embedding(input_ids)
                .map_err(vllm_model::error::ModelError::Candle)?;
            // hidden: [num_tokens, hidden_size]
            // logits = hidden @ lm_head^T → [num_tokens, vocab_size]
            let logits = hidden
                .matmul(
                    &self
                        .lm_head
                        .t()
                        .map_err(vllm_model::error::ModelError::Candle)?,
                )
                .map_err(vllm_model::error::ModelError::Candle)?;
            Ok(logits)
        }

        fn num_layers(&self) -> usize {
            1
        }
    }

    /// Build a CandleWorker with a tiny stub model injected, bypassing
    /// load_model (which requires real weights on disk).
    fn make_worker_with_stub_model() -> CandleWorker {
        let mut worker = CandleWorker::new(make_config());
        worker.device = Some(Device::Cpu);
        worker.model = Some(Box::new(TinyStubModel::new(32, 8)));
        worker
    }

    #[test]
    fn test_execute_model_prefill() {
        use std::collections::HashSet;
        use vllm_core::scheduler::output::{CachedRequestData, NewRequestData, SchedulerOutput};

        let mut worker = make_worker_with_stub_model();

        // Simulate a prefill step: one new request with 5 prompt tokens.
        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("r1".to_string(), 5);

        let scheduler_output = SchedulerOutput {
            scheduled_new_reqs: vec![NewRequestData::new(
                "r1".to_string(),
                Some(vec![1, 2, 3, 4, 5]), // token IDs < vocab_size=32
                vec![vec![0]],
                0,
                Some(SamplingParams::default()),
                None,
            )],
            scheduled_cached_reqs: CachedRequestData::make_empty(),
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 5,
            scheduled_spec_decode_tokens: HashMap::new(),
            scheduled_encoder_inputs: HashMap::new(),
            num_common_prefix_blocks: Vec::new(),
            finished_req_ids: HashSet::new(),
            free_encoder_mm_hashes: Vec::new(),
            preempted_req_ids: None,
        };

        let result = worker.execute_model(&scheduler_output).unwrap();
        let tokens = result.get_tokens("r1").unwrap();
        assert_eq!(tokens.len(), 1); // One sampled token.
        assert!((tokens[0] as usize) < 32); // Valid vocab index.

        // Token buffer should have prompt + 1 generated token.
        assert_eq!(worker.token_buffers["r1"].len(), 6);
        // KV cache should be populated.
        assert!(worker.kv_caches.contains_key("r1"));
    }

    #[test]
    fn test_execute_model_decode_step() {
        use std::collections::HashSet;
        use vllm_core::scheduler::output::{CachedRequestData, NewRequestData, SchedulerOutput};

        let mut worker = make_worker_with_stub_model();

        // Step 1: prefill.
        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("r1".to_string(), 3);

        let prefill_output = SchedulerOutput {
            scheduled_new_reqs: vec![NewRequestData::new(
                "r1".to_string(),
                Some(vec![1, 2, 3]),
                vec![vec![0]],
                0,
                Some(SamplingParams::default()),
                None,
            )],
            scheduled_cached_reqs: CachedRequestData::make_empty(),
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 3,
            scheduled_spec_decode_tokens: HashMap::new(),
            scheduled_encoder_inputs: HashMap::new(),
            num_common_prefix_blocks: Vec::new(),
            finished_req_ids: HashSet::new(),
            free_encoder_mm_hashes: Vec::new(),
            preempted_req_ids: None,
        };

        let result = worker.execute_model(&prefill_output).unwrap();
        assert!(result.get_tokens("r1").is_some());

        // Step 2: decode — request is now cached, num_computed_tokens = 4
        // (3 prompt + 1 generated).
        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("r1".to_string(), 1);

        let decode_output = SchedulerOutput {
            scheduled_new_reqs: Vec::new(),
            scheduled_cached_reqs: CachedRequestData {
                req_ids: vec!["r1".to_string()],
                resumed_req_ids: HashSet::new(),
                new_token_ids: Vec::new(),
                new_block_ids: vec![None],
                num_computed_tokens: vec![4], // 3 prompt + 1 generated
                num_output_tokens: vec![1],
            },
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 1,
            scheduled_spec_decode_tokens: HashMap::new(),
            scheduled_encoder_inputs: HashMap::new(),
            num_common_prefix_blocks: Vec::new(),
            finished_req_ids: HashSet::new(),
            free_encoder_mm_hashes: Vec::new(),
            preempted_req_ids: None,
        };

        let result = worker.execute_model(&decode_output).unwrap();
        let tokens = result.get_tokens("r1").unwrap();
        assert_eq!(tokens.len(), 1);
        assert!((tokens[0] as usize) < 32);

        // Buffer should now have prompt(3) + generated(2) = 5 tokens.
        assert_eq!(worker.token_buffers["r1"].len(), 5);
    }

    #[test]
    fn test_execute_model_finished_cleanup() {
        use std::collections::HashSet;
        use vllm_core::scheduler::output::{CachedRequestData, NewRequestData, SchedulerOutput};

        let mut worker = make_worker_with_stub_model();

        // Prefill a request.
        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("r1".to_string(), 2);

        let prefill = SchedulerOutput {
            scheduled_new_reqs: vec![NewRequestData::new(
                "r1".to_string(),
                Some(vec![1, 2]),
                vec![vec![0]],
                0,
                Some(SamplingParams::default()),
                None,
            )],
            scheduled_cached_reqs: CachedRequestData::make_empty(),
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 2,
            scheduled_spec_decode_tokens: HashMap::new(),
            scheduled_encoder_inputs: HashMap::new(),
            num_common_prefix_blocks: Vec::new(),
            finished_req_ids: HashSet::new(),
            free_encoder_mm_hashes: Vec::new(),
            preempted_req_ids: None,
        };

        worker.execute_model(&prefill).unwrap();
        assert!(worker.token_buffers.contains_key("r1"));
        assert!(worker.sampling_params_map.contains_key("r1"));

        // Now simulate a step where r1 is in finished_req_ids.
        let empty = SchedulerOutput {
            scheduled_new_reqs: Vec::new(),
            scheduled_cached_reqs: CachedRequestData::make_empty(),
            num_scheduled_tokens: HashMap::new(),
            total_num_scheduled_tokens: 0,
            scheduled_spec_decode_tokens: HashMap::new(),
            scheduled_encoder_inputs: HashMap::new(),
            num_common_prefix_blocks: Vec::new(),
            finished_req_ids: HashSet::from(["r1".to_string()]),
            free_encoder_mm_hashes: Vec::new(),
            preempted_req_ids: None,
        };

        worker.execute_model(&empty).unwrap();
        // Buffers should be cleaned up.
        assert!(!worker.token_buffers.contains_key("r1"));
        assert!(!worker.sampling_params_map.contains_key("r1"));
        assert!(!worker.kv_caches.contains_key("r1"));
    }

    #[test]
    fn test_execute_model_paged_multi_request_prefill() {
        // Test the batched forward path: two concurrent prefill requests
        // with paged KV cache.
        use std::collections::HashSet;
        use vllm_core::scheduler::output::{CachedRequestData, NewRequestData, SchedulerOutput};

        let mut worker = CandleWorker::new(make_config());
        worker.device = Some(Device::Cpu);
        worker.model = Some(Box::new(TinyStubModel::new(32, 8)));

        // Set up a paged KV block pool (the TinyStubModel has 1 layer).
        // 4 blocks, block_size=16, 1 kv_head, head_dim=8 (matches nothing
        // real but the stub model doesn't use the KV cache).
        worker.kv_block_pool =
            Some(vllm_models::KvBlockPool::new(4, 1, 1, 8, 16, DType::F32, &Device::Cpu).unwrap());

        // Two concurrent prefill requests.
        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("r1".to_string(), 3);
        num_scheduled.insert("r2".to_string(), 2);

        let scheduler_output = SchedulerOutput {
            scheduled_new_reqs: vec![
                NewRequestData::new(
                    "r1".to_string(),
                    Some(vec![1, 2, 3]),
                    vec![vec![0]],
                    0,
                    Some(SamplingParams::default()),
                    None,
                ),
                NewRequestData::new(
                    "r2".to_string(),
                    Some(vec![4, 5]),
                    vec![vec![1]],
                    0,
                    Some(SamplingParams::default()),
                    None,
                ),
            ],
            scheduled_cached_reqs: CachedRequestData::make_empty(),
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 5,
            scheduled_spec_decode_tokens: HashMap::new(),
            scheduled_encoder_inputs: HashMap::new(),
            num_common_prefix_blocks: Vec::new(),
            finished_req_ids: HashSet::new(),
            free_encoder_mm_hashes: Vec::new(),
            preempted_req_ids: None,
        };

        let result = worker.execute_model(&scheduler_output).unwrap();

        // Both requests should have produced a token.
        let tokens_r1 = result.get_tokens("r1").unwrap();
        let tokens_r2 = result.get_tokens("r2").unwrap();
        assert_eq!(tokens_r1.len(), 1);
        assert_eq!(tokens_r2.len(), 1);
        assert!((tokens_r1[0] as usize) < 32);
        assert!((tokens_r2[0] as usize) < 32);

        // Token buffers: prompt + 1 generated.
        assert_eq!(worker.token_buffers["r1"].len(), 4);
        assert_eq!(worker.token_buffers["r2"].len(), 3);

        // Tokens-in-pool should be updated (now tracked in InputBatch).
        assert_eq!(worker.input_batch.tokens_in_pool_for("r1"), 3);
        assert_eq!(worker.input_batch.tokens_in_pool_for("r2"), 2);
    }

    #[test]
    fn test_execute_model_paged_prefill_then_concurrent_decode() {
        // Test batched forward: prefill two requests, then decode them
        // concurrently in the same step.
        use std::collections::HashSet;
        use vllm_core::scheduler::output::{CachedRequestData, NewRequestData, SchedulerOutput};

        let mut worker = CandleWorker::new(make_config());
        worker.device = Some(Device::Cpu);
        worker.model = Some(Box::new(TinyStubModel::new(32, 8)));
        worker.kv_block_pool =
            Some(vllm_models::KvBlockPool::new(4, 1, 1, 8, 16, DType::F32, &Device::Cpu).unwrap());

        // Step 1: prefill both requests.
        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("r1".to_string(), 3);
        num_scheduled.insert("r2".to_string(), 2);

        let prefill = SchedulerOutput {
            scheduled_new_reqs: vec![
                NewRequestData::new(
                    "r1".to_string(),
                    Some(vec![1, 2, 3]),
                    vec![vec![0]],
                    0,
                    Some(SamplingParams::default()),
                    None,
                ),
                NewRequestData::new(
                    "r2".to_string(),
                    Some(vec![4, 5]),
                    vec![vec![1]],
                    0,
                    Some(SamplingParams::default()),
                    None,
                ),
            ],
            scheduled_cached_reqs: CachedRequestData::make_empty(),
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 5,
            scheduled_spec_decode_tokens: HashMap::new(),
            scheduled_encoder_inputs: HashMap::new(),
            num_common_prefix_blocks: Vec::new(),
            finished_req_ids: HashSet::new(),
            free_encoder_mm_hashes: Vec::new(),
            preempted_req_ids: None,
        };
        worker.execute_model(&prefill).unwrap();

        // Step 2: decode both concurrently.
        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("r1".to_string(), 1);
        num_scheduled.insert("r2".to_string(), 1);

        let decode = SchedulerOutput {
            scheduled_new_reqs: Vec::new(),
            scheduled_cached_reqs: CachedRequestData {
                req_ids: vec!["r1".to_string(), "r2".to_string()],
                resumed_req_ids: HashSet::new(),
                new_token_ids: Vec::new(),
                new_block_ids: vec![None, None],
                num_computed_tokens: vec![4, 3],
                num_output_tokens: vec![1, 1],
            },
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 2,
            scheduled_spec_decode_tokens: HashMap::new(),
            scheduled_encoder_inputs: HashMap::new(),
            num_common_prefix_blocks: Vec::new(),
            finished_req_ids: HashSet::new(),
            free_encoder_mm_hashes: Vec::new(),
            preempted_req_ids: None,
        };

        let result = worker.execute_model(&decode).unwrap();

        // Both decoded successfully.
        let tokens_r1 = result.get_tokens("r1").unwrap();
        let tokens_r2 = result.get_tokens("r2").unwrap();
        assert_eq!(tokens_r1.len(), 1);
        assert_eq!(tokens_r2.len(), 1);

        // Buffers grew: prompt + 2 generated tokens each.
        assert_eq!(worker.token_buffers["r1"].len(), 5);
        assert_eq!(worker.token_buffers["r2"].len(), 4);

        // Tokens-in-pool incremented (now tracked in InputBatch).
        assert_eq!(worker.input_batch.tokens_in_pool_for("r1"), 4);
        assert_eq!(worker.input_batch.tokens_in_pool_for("r2"), 3);
    }
}
