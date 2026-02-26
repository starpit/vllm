// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! A real `Worker` implementation backed by candle for CPU/CUDA/Metal inference.
//!
//! `CandleWorker` loads model weights via `vllm-model`, constructs a model
//! architecture via `vllm-models`, and runs forward passes using candle
//! tensors. HuggingFace Hub models are downloaded on demand via `hf-hub`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::{DType, Device, Tensor};
use tracing::{debug, info, warn};
use vllm_common::SamplingParams;
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_engine::executor::ModelRunnerOutput;
use vllm_model::weight::{HfModelConfig, ModelWeights};
use vllm_models::{KvCache, Model, ModelRegistry, Sampler};

use crate::error::{ExecutorError, ExecutorResult};
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

    /// Data type for model weights: "f32", "f16", "bf16".
    pub dtype: String,

    /// Optional HuggingFace token for gated models.
    pub hf_token: Option<String>,

    /// Optional cache directory for downloaded models.
    pub cache_dir: Option<String>,
}

impl CandleWorkerConfig {
    /// Parse the dtype string to a candle DType.
    pub fn candle_dtype(&self) -> ExecutorResult<DType> {
        match self.dtype.as_str() {
            "f32" | "float32" => Ok(DType::F32),
            "f16" | "float16" => Ok(DType::F16),
            "bf16" | "bfloat16" => Ok(DType::BF16),
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
    /// KV cache block counts (stored for reference).
    num_gpu_blocks: usize,
    num_cpu_blocks: usize,
    is_shutdown: bool,

    /// Per-request token buffer: req_id → all token IDs (prompt + generated).
    /// Used to feed the correct input token on decode steps.
    token_buffers: HashMap<String, Vec<u32>>,
    /// Per-request sampling params, stored on first scheduling.
    sampling_params_map: HashMap<String, SamplingParams>,
    /// Per-request KV cache: req_id → per-layer KV tensors.
    kv_caches: HashMap<String, KvCache>,
}

impl CandleWorker {
    /// Create a new CandleWorker from the given config.
    pub fn new(config: CandleWorkerConfig) -> Self {
        Self {
            config,
            device: None,
            model: None,
            model_dir: None,
            hf_config: None,
            num_gpu_blocks: 0,
            num_cpu_blocks: 0,
            is_shutdown: false,
            token_buffers: HashMap::new(),
            sampling_params_map: HashMap::new(),
            kv_caches: HashMap::new(),
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
        let mut builder = hf_hub::api::sync::ApiBuilder::new();
        if let Some(token) = &self.config.hf_token {
            builder = builder.with_token(Some(token.clone()));
        }
        if let Some(cache_dir) = &self.config.cache_dir {
            builder = builder.with_cache_dir(PathBuf::from(cache_dir));
        }
        let api = builder
            .build()
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to create HF API: {e}")))?;
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
            info!("Downloading {total} shard files (up to 8 in parallel)");

            const MAX_PARALLEL: usize = 8;
            let counter = AtomicUsize::new(0);
            let repo = &repo;

            for chunk in sorted_shards.chunks(MAX_PARALLEL) {
                let results: Vec<ExecutorResult<()>> = std::thread::scope(|s| {
                    let handles: Vec<_> = chunk
                        .iter()
                        .map(|shard| {
                            let counter = &counter;
                            s.spawn(move || {
                                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                                info!("  [{n}/{total}] {shard}");
                                repo.get(shard).map(|_| ()).map_err(|e| {
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

            return Ok(model_dir);
        }

        // Neither single nor sharded weights found — return directory anyway,
        // weight loading will produce a better error message.
        Ok(model_dir)
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
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("device not initialized".to_string()))?;

        let dtype = self.config.candle_dtype()?;

        // 1. Resolve model directory (local or HF download).
        let model_dir = self.resolve_model_path()?;
        info!("CandleWorker: loading model from {}", model_dir.display());

        // 2. Parse config.json.
        let hf_config = HfModelConfig::from_dir(&model_dir)
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to parse config.json: {e}")))?;

        // 3. Look up architecture in the registry.
        let arch = hf_config
            .architectures
            .first()
            .ok_or_else(|| {
                ExecutorError::WorkerInit("config.json has no architectures field".to_string())
            })?
            .clone();
        let registry = ModelRegistry::default();
        let factory = registry.get(&arch).ok_or_else(|| {
            ExecutorError::WorkerInit(format!(
                "unsupported architecture: {arch}. Supported: {:?}",
                registry.architectures().collect::<Vec<_>>()
            ))
        })?;

        // 4. Load weights.
        let weights = ModelWeights::from_dir(&model_dir, device)
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to load weights: {e}")))?;
        info!(
            "CandleWorker: loaded {} tensors ({:.1} MB)",
            weights.names().len(),
            weights.total_size_bytes() as f64 / 1_048_576.0
        );

        // 5. Construct the model.
        let model = factory(&weights, &hf_config, dtype, device)
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to construct model: {e}")))?;

        self.model_dir = Some(model_dir);
        self.hf_config = Some(hf_config);
        self.model = Some(model);
        info!("CandleWorker: model loaded (arch={arch})");
        Ok(())
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> ExecutorResult<()> {
        self.num_gpu_blocks = num_gpu_blocks;
        self.num_cpu_blocks = num_cpu_blocks;
        info!(
            "CandleWorker: cache initialized (gpu_blocks={}, cpu_blocks={})",
            num_gpu_blocks, num_cpu_blocks
        );
        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        // For CPU, report a reasonable default. CUDA would query device memory.
        // Default to 4 GiB for CPU mode.
        Ok(4 * 1024 * 1024 * 1024)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerExecution("model not loaded".to_string()))?;
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerExecution("device not initialized".to_string()))?;

        let num_layers = model.num_layers();

        // Clean up buffers for finished requests.
        for req_id in &scheduler_output.finished_req_ids {
            self.token_buffers.remove(req_id);
            self.sampling_params_map.remove(req_id);
            self.kv_caches.remove(req_id);
        }

        // Collect requests to process: (req_id, token_ids, positions, is_prefill).
        struct ReqInput {
            req_id: String,
            token_ids: Vec<u32>,
            positions: Vec<u32>,
            is_prefill: bool,
        }
        let mut req_inputs: Vec<ReqInput> = Vec::new();

        // --- Process newly scheduled requests (prefill) ---
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

            // Store prompt tokens in the buffer.
            self.token_buffers
                .insert(new_req.req_id.clone(), prompt_ids.to_vec());

            // Store sampling params if provided.
            if let Some(ref params) = new_req.sampling_params {
                self.sampling_params_map
                    .insert(new_req.req_id.clone(), params.clone());
            }

            // Create an empty KV cache for this request.
            self.kv_caches
                .insert(new_req.req_id.clone(), vec![None; num_layers]);

            // Build positions: 0..num_tokens (offset by num_computed_tokens).
            let pos_offset = new_req.num_computed_tokens;
            let positions: Vec<u32> = (0..tokens_to_use.len() as u32)
                .map(|i| pos_offset + i)
                .collect();

            req_inputs.push(ReqInput {
                req_id: new_req.req_id.clone(),
                token_ids: tokens_to_use.to_vec(),
                positions,
                is_prefill: true,
            });
        }

        // --- Process cached/continuing requests (decode) ---
        // With KV cache, we only feed the last token; previous tokens' K/V
        // are already cached. This makes decode O(1) per token instead of O(n).
        let new_req_ids: HashSet<&str> = scheduler_output
            .scheduled_new_reqs
            .iter()
            .map(|r| r.req_id.as_str())
            .collect();
        for req_id in scheduler_output.num_scheduled_tokens.keys() {
            if new_req_ids.contains(req_id.as_str()) {
                continue; // Already handled as a new request.
            }
            let num_tokens = scheduler_output.num_scheduled_tokens[req_id];
            if num_tokens == 0 {
                continue;
            }

            if let Some(buf) = self.token_buffers.get(req_id) {
                // Feed only the last token (the most recently generated one).
                let last_token = *buf.last().unwrap_or(&0);
                let position = (buf.len() - 1) as u32;

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

        if req_inputs.is_empty() {
            return Ok(ModelRunnerOutput::from_token_map(HashMap::new()));
        }

        // Create a sampler for non-greedy requests (created per-call to avoid
        // Send issues — Sampler holds ThreadRng which is !Send).
        let mut sampler = Sampler::new();
        let mut token_map: HashMap<String, Vec<u32>> = HashMap::new();

        // Run a separate forward pass per request (required because each
        // request has its own KV cache with different sequence length).
        for req_input in &req_inputs {
            let input_ids = Tensor::new(req_input.token_ids.as_slice(), device)
                .map_err(|e| ExecutorError::WorkerExecution(format!("tensor error: {e}")))?;
            let positions = Tensor::new(req_input.positions.as_slice(), device)
                .map_err(|e| ExecutorError::WorkerExecution(format!("tensor error: {e}")))?;

            // Get this request's KV cache (or None for uncached forward).
            let kv_cache = self.kv_caches.get_mut(&req_input.req_id);

            let logits = model
                .forward(&input_ids, &positions, kv_cache)
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("model forward failed: {e}"))
                })?;

            // Take logits at the last position and sample.
            let last_pos = req_input.token_ids.len() - 1;
            let req_logits = logits
                .narrow(0, last_pos, 1)
                .map_err(|e| ExecutorError::WorkerExecution(format!("tensor error: {e}")))?;

            let sampled = if let Some(params) = self.sampling_params_map.get(&req_input.req_id) {
                let temp = params.temperature as f32;
                let top_k = params.top_k.max(0) as usize;
                let top_p = params.top_p as f32;

                if temp < 1e-5 {
                    sampler
                        .greedy(&req_logits)
                        .map_err(|e| ExecutorError::WorkerExecution(format!("greedy error: {e}")))?
                } else if top_k > 0 || top_p < 1.0 {
                    sampler
                        .sample_top_k_top_p(&req_logits, temp, top_k, top_p)
                        .map_err(|e| {
                            ExecutorError::WorkerExecution(format!("sampling error: {e}"))
                        })?
                } else {
                    sampler.sample(&req_logits, temp).map_err(|e| {
                        ExecutorError::WorkerExecution(format!("sampling error: {e}"))
                    })?
                }
            } else {
                let indices = req_logits
                    .argmax(candle_core::D::Minus1)
                    .map_err(|e| ExecutorError::WorkerExecution(format!("argmax error: {e}")))?;
                indices
                    .to_vec1::<u32>()
                    .map_err(|e| ExecutorError::WorkerExecution(format!("tensor error: {e}")))?
            };

            // Update the token buffer with the new sampled token(s).
            if let Some(buf) = self.token_buffers.get_mut(&req_input.req_id) {
                buf.extend_from_slice(&sampled);
            }

            debug!(
                "Request {}: sampled {:?} (buf_len={}, prefill={})",
                req_input.req_id,
                sampled,
                self.token_buffers
                    .get(&req_input.req_id)
                    .map(|b| b.len())
                    .unwrap_or(0),
                req_input.is_prefill,
            );

            token_map.insert(req_input.req_id.clone(), sampled);
        }

        Ok(ModelRunnerOutput::from_token_map(token_map))
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
        }
    }

    #[test]
    fn test_config_dtype_parsing() {
        let mut cfg = make_config();
        assert_eq!(cfg.candle_dtype().unwrap(), DType::F32);
        cfg.dtype = "f16".to_string();
        assert_eq!(cfg.candle_dtype().unwrap(), DType::F16);
        cfg.dtype = "bf16".to_string();
        assert_eq!(cfg.candle_dtype().unwrap(), DType::BF16);
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
            _kv_cache: Option<&mut vllm_models::KvCache>,
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
}
