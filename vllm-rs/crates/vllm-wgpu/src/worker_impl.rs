// SPDX-License-Identifier: Apache-2.0
//! WebGPU worker implementing the `Worker` trait from `vllm-executor`.
//!
//! Bridges the wgpu inference engine into the vLLM scheduler/engine pipeline,
//! enabling `vllm serve --device wgpu` and `vllm chat --device wgpu`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tracing::{debug, info};

use vllm_common::SamplingParams;
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_engine::executor::ModelRunnerOutput;
use vllm_executor::error::{ExecutorError, ExecutorResult};
use vllm_executor::worker::Worker;
use vllm_model::weight::HfModelConfig;

use crate::WgpuDevice;
use crate::model::{ModelConfig, WgpuWorker};

// ---------------------------------------------------------------------------
// WgpuBackendWorker
// ---------------------------------------------------------------------------

/// A `Worker` implementation backed by the WebGPU inference engine.
///
/// Wraps [`WgpuWorker`] and translates between the scheduler's
/// `SchedulerOutput` / `ModelRunnerOutput` protocol and wgpu's
/// per-request `forward_one` / `forward_batch` API.
pub struct WgpuBackendWorker {
    model_path: String,
    inner: Option<WgpuWorker>,
    hf_config: Option<HfModelConfig>,
    model_dir: Option<PathBuf>,
    preloaded_tokenizer: Option<tokenizers::Tokenizer>,
    resolved_architecture: Option<String>,

    /// Per-request token buffer: req_id -> all token IDs (prompt + generated).
    token_buffers: HashMap<String, Vec<u32>>,
    /// Per-request sampling params.
    sampling_params_map: HashMap<String, SamplingParams>,
    /// Per-request KV cache position (how many tokens have been forwarded).
    kv_positions: HashMap<String, usize>,

    is_shutdown: bool,
}

impl WgpuBackendWorker {
    /// Create a new worker targeting the given model path or HuggingFace ID.
    pub fn new(model_path: String) -> Self {
        Self {
            model_path,
            inner: None,
            hf_config: None,
            model_dir: None,
            preloaded_tokenizer: None,
            resolved_architecture: None,
            token_buffers: HashMap::new(),
            sampling_params_map: HashMap::new(),
            kv_positions: HashMap::new(),
            is_shutdown: false,
        }
    }

    /// Access the parsed HuggingFace config (available after `load_model`).
    pub fn hf_config(&self) -> Option<&HfModelConfig> {
        self.hf_config.as_ref()
    }

    /// Access the resolved model directory (available after `load_model`).
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Worker trait
// ---------------------------------------------------------------------------

impl Worker for WgpuBackendWorker {
    fn init_device(&mut self) -> ExecutorResult<()> {
        // wgpu device creation is deferred to load_model (from_pretrained
        // creates the device internally). Nothing to do here.
        Ok(())
    }

    fn load_model(&mut self) -> ExecutorResult<()> {
        info!("Loading model on WebGPU: {}", self.model_path);

        let device = pollster::block_on(WgpuDevice::new())
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to create wgpu device: {e}")))?;

        let model_path = self.model_path.clone();
        let (worker, config, tokenizer) = WgpuWorker::from_pretrained(device, &model_path)
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to load model: {e}")))?;

        // Get model_dir from the worker (set during from_pretrained).
        let model_dir = worker.model_dir().map(|p| p.to_path_buf());

        // Build HfModelConfig from the wgpu ModelConfig + config.json extras.
        let hf_config = wgpu_config_to_hf(&config, model_dir.as_deref());

        // Detect architecture from config.json if available.
        let architecture = model_dir.as_ref().and_then(|dir| {
            let config_path = dir.join("config.json");
            let text = std::fs::read_to_string(&config_path).ok()?;
            let val: serde_json::Value = serde_json::from_str(&text).ok()?;
            val.get("architectures")
                .and_then(|a| a.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });

        self.preloaded_tokenizer = Some(tokenizer);
        self.inner = Some(worker);
        self.hf_config = Some(hf_config);
        self.model_dir = model_dir;
        self.resolved_architecture = architecture;

        info!("WebGPU model loaded successfully");
        Ok(())
    }

    fn initialize_cache(
        &mut self,
        _num_gpu_blocks: usize,
        _num_cpu_blocks: usize,
    ) -> ExecutorResult<()> {
        // WgpuWorker manages its own per-layer contiguous KV cache internally.
        // The scheduler's paged block system doesn't map to wgpu's cache layout.
        // No-op — cache is already allocated in WgpuWorker::new().
        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        // wgpu doesn't expose a direct GPU memory query.
        // Return a conservative 2 GB estimate — this is used to compute the
        // number of KV cache blocks for the scheduler. Since wgpu manages its
        // own cache internally, the exact value doesn't matter much.
        const WGPU_AVAILABLE_MEMORY: usize = 2 * 1024 * 1024 * 1024;
        Ok(WGPU_AVAILABLE_MEMORY)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        let worker = self
            .inner
            .as_mut()
            .ok_or_else(|| ExecutorError::WorkerExecution("wgpu model not loaded".to_string()))?;

        // 1. Clean up finished requests.
        for req_id in &scheduler_output.finished_req_ids {
            self.token_buffers.remove(req_id);
            self.sampling_params_map.remove(req_id);
            self.kv_positions.remove(req_id);
            // Reset KV cache when a request finishes so the next request
            // gets a clean slate. (Single-request limitation for now.)
            worker.reset_kv();
        }

        // 2. Register new requests.
        for new_req in &scheduler_output.scheduled_new_reqs {
            let prompt_ids = new_req.prompt_token_ids.as_deref().unwrap_or(&[]);
            self.token_buffers
                .insert(new_req.req_id.clone(), prompt_ids.to_vec());
            self.kv_positions.insert(new_req.req_id.clone(), 0);
            if let Some(ref params) = new_req.sampling_params {
                self.sampling_params_map
                    .insert(new_req.req_id.clone(), params.clone());
            }
        }

        // 3. Process each request.
        let mut token_map: HashMap<String, Vec<u32>> = HashMap::new();

        for (req_id, &num_tokens) in &scheduler_output.num_scheduled_tokens {
            if num_tokens == 0 {
                continue;
            }

            let kv_pos = self.kv_positions.get(req_id).copied().unwrap_or(0);
            let tokens = self.token_buffers.get(req_id).cloned().unwrap_or_default();

            let next_token = if kv_pos == 0 && tokens.len() > 1 {
                // Prefill: forward the prompt tokens.
                let prompt_tokens = &tokens;
                debug!(
                    "wgpu prefill req={} tokens={} pos=0",
                    req_id,
                    prompt_tokens.len()
                );
                let token_id =
                    pollster::block_on(worker.forward_batch(prompt_tokens, 0)).map_err(|e| {
                        ExecutorError::WorkerExecution(format!(
                            "wgpu prefill failed for {req_id}: {e}"
                        ))
                    })?;
                // Update KV position to include all prompt tokens.
                self.kv_positions.insert(req_id.clone(), tokens.len());
                token_id
            } else {
                // Decode: forward the last generated token.
                let last_token = tokens.last().copied().unwrap_or(0);
                debug!(
                    "wgpu decode req={} token={} pos={}",
                    req_id, last_token, kv_pos
                );
                let token_id =
                    pollster::block_on(worker.forward_one(last_token, kv_pos)).map_err(|e| {
                        ExecutorError::WorkerExecution(format!(
                            "wgpu decode failed for {req_id}: {e}"
                        ))
                    })?;
                self.kv_positions.insert(req_id.clone(), kv_pos + 1);
                token_id
            };

            // Append generated token to the buffer.
            if let Some(buf) = self.token_buffers.get_mut(req_id) {
                buf.push(next_token);
            }

            token_map.insert(req_id.clone(), vec![next_token]);
        }

        Ok(ModelRunnerOutput::from_token_map(token_map))
    }

    fn take_preloaded_tokenizer(&mut self) -> Option<tokenizers::Tokenizer> {
        self.preloaded_tokenizer.take()
    }

    fn architecture(&self) -> Option<String> {
        self.resolved_architecture.clone()
    }

    fn shutdown(&mut self) {
        self.is_shutdown = true;
        self.inner = None;
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

/// Convert a wgpu `ModelConfig` to an `HfModelConfig` for the init stack.
fn wgpu_config_to_hf(config: &ModelConfig, model_dir: Option<&Path>) -> HfModelConfig {
    // Try to read the full config.json for extra fields (eos_token_id, etc.)
    let (architectures, extra) = model_dir
        .and_then(|dir| {
            let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
            let val: serde_json::Value = serde_json::from_str(&text).ok()?;
            let archs = val
                .get("architectures")
                .and_then(|a| a.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let extra = val
                .as_object()
                .map(|m| m.clone().into_iter().collect())
                .unwrap_or_default();
            Some((archs, extra))
        })
        .unwrap_or_default();

    HfModelConfig {
        hidden_size: Some(config.hidden_size),
        num_hidden_layers: Some(config.num_hidden_layers),
        num_attention_heads: Some(config.num_attention_heads),
        num_key_value_heads: if config.num_key_value_heads > 0 {
            Some(config.num_key_value_heads)
        } else {
            Some(config.num_attention_heads)
        },
        max_position_embeddings: Some(config.max_position_embeddings),
        architectures,
        extra,
        ..Default::default()
    }
}
