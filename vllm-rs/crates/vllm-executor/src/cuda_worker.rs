// SPDX-License-Identifier: Apache-2.0
//! `CudaWorker`: a `Worker` implementation using the vllm-cuda backend.
//!
//! Replaces candle as the GPU runtime with `GpuTensor`/`GpuDevice`/`ScratchArena`
//! for zero-allocation inference. Uses the same paged FlashAttention-2 kernels,
//! but through raw FFI instead of candle CustomOps.
//!
//! This worker is gated behind the `cuda-backend` feature flag.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tracing::{info, warn};
use vllm_common::SamplingParams;
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_cuda::device::GpuDevice;
use vllm_cuda::driver;
use vllm_cuda::dtype::DType as GpuDType;
use vllm_cuda::kv_cache::KvCachePool;
use vllm_cuda::tensor::GpuTensor;
use vllm_cuda::weights::GpuWeights;
use vllm_engine::executor::ModelRunnerOutput;
use vllm_model::weight::HfModelConfig;
use vllm_models::Sampler;

use crate::error::{ExecutorError, ExecutorResult};
use crate::input_batch::InputBatch;
use crate::worker::Worker;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a `CudaWorker`.
#[derive(Debug, Clone)]
pub struct CudaWorkerConfig {
    /// Path to a local model directory, or a HuggingFace model ID.
    pub model_path: String,
    /// Data type for model weights: "auto", "f16", "bf16".
    pub dtype: String,
    /// Optional HuggingFace token for gated models.
    pub hf_token: Option<String>,
    /// KV cache block size in tokens (must match the scheduler's block size).
    pub block_size: usize,
    /// GPU device index.
    pub device_id: i32,
}

// ---------------------------------------------------------------------------
// Model enum (dispatches to LLaMA / Qwen2 / Gemma2)
// ---------------------------------------------------------------------------

/// Supported model architectures in the vllm-cuda backend.
enum CudaModel {
    Llama(vllm_cuda::model::llama::LlamaForCausalLM),
    Qwen2(vllm_cuda::model::qwen2::Qwen2ForCausalLM),
    Gemma2(vllm_cuda::model::gemma2::Gemma2ForCausalLM),
}

impl CudaModel {
    fn num_layers(&self) -> usize {
        match self {
            Self::Llama(m) => m.model.layers.len(),
            Self::Qwen2(m) => m.0.model.layers.len(),
            Self::Gemma2(m) => m.model.layers.len(),
        }
    }

    fn num_kv_heads(&self) -> usize {
        match self {
            Self::Llama(m) => m.model.layers[0].self_attn.num_kv_heads,
            Self::Qwen2(m) => m.0.model.layers[0].self_attn.num_kv_heads,
            Self::Gemma2(m) => m.model.layers[0].self_attn.num_kv_heads,
        }
    }

    fn head_dim(&self) -> usize {
        match self {
            Self::Llama(m) => m.model.layers[0].self_attn.head_dim,
            Self::Qwen2(m) => m.0.model.layers[0].self_attn.head_dim,
            Self::Gemma2(m) => m.model.layers[0].self_attn.head_dim,
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Llama(m) => m.lm_head.out_features(),
            Self::Qwen2(m) => m.0.lm_head.out_features(),
            Self::Gemma2(m) => m.lm_head.out_features(),
        }
    }

    /// Run forward pass, returning logits `[num_reqs, vocab_size]`.
    ///
    /// # Safety
    /// All GpuTensors must be valid. CUDA context must be current.
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward(
        &self,
        input_ids: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        cu_seqlens_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
    ) -> GpuTensor {
        match self {
            Self::Llama(m) => m.forward(
                input_ids, positions, slot_mapping,
                cu_seqlens_q, cu_seqlens_k, block_table,
                max_seqlen_q, max_seqlen_k, kv_cache, device,
            ),
            Self::Qwen2(m) => m.forward(
                input_ids, positions, slot_mapping,
                cu_seqlens_q, cu_seqlens_k, block_table,
                max_seqlen_q, max_seqlen_k, kv_cache, device,
            ),
            Self::Gemma2(m) => m.forward(
                input_ids, positions, slot_mapping,
                cu_seqlens_q, cu_seqlens_k, block_table,
                max_seqlen_q, max_seqlen_k, kv_cache, device,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Config helpers: HfModelConfig → vllm-cuda configs
// ---------------------------------------------------------------------------

fn llama_config_from_hf(hf: &HfModelConfig) -> ExecutorResult<vllm_cuda::model::llama::LlamaConfig> {
    let hidden_size = hf.hidden_size.ok_or_else(|| {
        ExecutorError::WorkerInit("missing hidden_size in config.json".into())
    })?;
    let num_attention_heads = hf.num_attention_heads.ok_or_else(|| {
        ExecutorError::WorkerInit("missing num_attention_heads".into())
    })?;
    let num_kv_heads = hf.num_key_value_heads.unwrap_or(num_attention_heads);
    let head_dim = hf.head_dim.unwrap_or(hidden_size / num_attention_heads);

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
    })
}

fn gemma2_config_from_hf(hf: &HfModelConfig) -> ExecutorResult<vllm_cuda::model::gemma2::Gemma2Config> {
    let hidden_size = hf.hidden_size.ok_or_else(|| {
        ExecutorError::WorkerInit("missing hidden_size".into())
    })?;
    let num_attention_heads = hf.num_attention_heads.ok_or_else(|| {
        ExecutorError::WorkerInit("missing num_attention_heads".into())
    })?;
    let num_kv_heads = hf.num_key_value_heads.unwrap_or(num_attention_heads);
    let num_hidden_layers = hf.num_hidden_layers.unwrap_or(26);
    let head_dim = hf.head_dim.unwrap_or(hidden_size / num_attention_heads);

    let query_pre_attn_scalar = hf.extra.get("query_pre_attn_scalar")
        .and_then(|v| v.as_f64())
        .unwrap_or(head_dim as f64);
    let attn_logit_softcapping = hf.extra.get("attn_logit_softcapping")
        .and_then(|v| v.as_f64());
    let final_logit_softcapping = hf.extra.get("final_logit_softcapping")
        .and_then(|v| v.as_f64());
    let sliding_window = hf.extra.get("sliding_window")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    // Gemma2 alternates sliding/full attention: even layers = sliding, odd = full.
    // Or it may be specified via sliding_window_pattern in config.
    let layer_is_sliding: Vec<bool> = (0..num_hidden_layers)
        .map(|i| i % 2 == 0)
        .collect();

    Ok(vllm_cuda::model::gemma2::Gemma2Config {
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        num_hidden_layers,
        intermediate_size: hf.intermediate_size.unwrap_or(hidden_size * 4),
        vocab_size: hf.vocab_size.unwrap_or(256000),
        max_position_embeddings: hf.max_position_embeddings.unwrap_or(8192),
        rms_norm_eps: hf.rms_norm_eps.unwrap_or(1e-6) as f32,
        rope_theta: hf.rope_theta.unwrap_or(10000.0),
        head_dim,
        query_pre_attn_scalar,
        attn_logit_softcapping,
        final_logit_softcapping,
        tie_word_embeddings: hf.tie_word_embeddings.unwrap_or(true),
        layer_is_sliding,
        sliding_window,
    })
}

// ---------------------------------------------------------------------------
// CudaWorker
// ---------------------------------------------------------------------------

/// A worker backed by the vllm-cuda runtime for zero-allocation GPU inference.
pub struct CudaWorker {
    config: CudaWorkerConfig,
    device: Option<GpuDevice>,
    model: Option<CudaModel>,
    kv_cache: Option<KvCachePool>,
    model_dir: Option<PathBuf>,
    hf_config: Option<HfModelConfig>,
    model_dtype: GpuDType,
    resolved_architecture: Option<String>,
    is_shutdown: bool,

    // Per-request state (mirrors CandleWorker).
    token_buffers: HashMap<String, Vec<u32>>,
    sampling_params_map: HashMap<String, SamplingParams>,
    input_batch: InputBatch,
    preloaded_tokenizer: Option<tokenizers::Tokenizer>,
}

unsafe impl Send for CudaWorker {}

impl CudaWorker {
    pub fn new(config: CudaWorkerConfig) -> Self {
        Self {
            config,
            device: None,
            model: None,
            kv_cache: None,
            model_dir: None,
            hf_config: None,
            model_dtype: GpuDType::BF16,
            resolved_architecture: None,
            is_shutdown: false,
            token_buffers: HashMap::new(),
            sampling_params_map: HashMap::new(),
            input_batch: InputBatch::new(),
            preloaded_tokenizer: None,
        }
    }

    /// Expose the HF config after load_model.
    pub fn hf_config(&self) -> Option<&HfModelConfig> {
        self.hf_config.as_ref()
    }

    /// Expose the model directory after load_model.
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }

    /// Resolved dtype as candle DType (for init.rs compatibility).
    pub fn resolved_candle_dtype(&self) -> candle_core::DType {
        match self.model_dtype {
            GpuDType::F16 => candle_core::DType::F16,
            GpuDType::BF16 => candle_core::DType::BF16,
            GpuDType::F32 => candle_core::DType::F32,
            _ => candle_core::DType::BF16,
        }
    }

    /// Resolve model path: local dir or HF download.
    fn resolve_model_path(&self) -> ExecutorResult<PathBuf> {
        let path = Path::new(&self.config.model_path);
        if path.is_dir() {
            return Ok(path.to_path_buf());
        }

        info!(
            "Downloading model from HuggingFace Hub: {}",
            self.config.model_path
        );
        let mut builder = hf_hub::api::sync::ApiBuilder::new();
        if let Some(ref token) = self.config.hf_token {
            builder = builder.with_token(Some(token.clone()));
        }
        let api = builder.build().map_err(|e| {
            ExecutorError::WorkerInit(format!("failed to build HF API: {e}"))
        })?;
        let repo = api.model(self.config.model_path.clone());

        let config_path = repo.get("config.json").map_err(|e| {
            ExecutorError::WorkerInit(format!("failed to download config.json: {e}"))
        })?;
        let model_dir = config_path.parent().unwrap().to_path_buf();

        // Best-effort tokenizer download.
        let _ = repo.get("tokenizer.json");
        let _ = repo.get("tokenizer_config.json");

        // Download weight files.
        if repo.get("model.safetensors").is_ok() {
            return Ok(model_dir);
        }
        if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            let index = vllm_model::weight::SafeTensorsIndex::from_file(&index_path)
                .map_err(|e| ExecutorError::WorkerInit(format!("failed to parse index: {e}")))?;
            for shard in index.shard_files() {
                repo.get(&shard).map_err(|e| {
                    ExecutorError::WorkerInit(format!("failed to download {shard}: {e}"))
                })?;
            }
            return Ok(model_dir);
        }

        Err(ExecutorError::WorkerInit(format!(
            "no safetensors weights found for {}",
            self.config.model_path
        )))
    }

    /// H2D copy a u32 slice into an arena-allocated GpuTensor.
    unsafe fn h2d_u32(
        data: &[u32],
        device: &mut GpuDevice,
    ) -> ExecutorResult<GpuTensor> {
        let t = device.arena.alloc(&[data.len()], GpuDType::U32);
        driver::memcpy_htod_async(
            t.raw_ptr(),
            data.as_ptr() as *const u8,
            data.len() * 4,
            device.compute_stream,
        )
        .map_err(|e| ExecutorError::WorkerExecution(format!("H2D u32: {e}")))?;
        Ok(t)
    }

    /// H2D copy an i64 slice into an arena-allocated GpuTensor.
    unsafe fn h2d_i64(
        data: &[i64],
        device: &mut GpuDevice,
    ) -> ExecutorResult<GpuTensor> {
        let t = device.arena.alloc(&[data.len()], GpuDType::I64);
        driver::memcpy_htod_async(
            t.raw_ptr(),
            data.as_ptr() as *const u8,
            data.len() * 8,
            device.compute_stream,
        )
        .map_err(|e| ExecutorError::WorkerExecution(format!("H2D i64: {e}")))?;
        Ok(t)
    }

    /// Build attention metadata tensors from `AttentionMetadata`.
    ///
    /// Returns `(slot_mapping, cu_seqlens_q, cu_seqlens_k, block_table, max_seqlen_q, max_seqlen_k)`.
    unsafe fn build_attention_tensors(
        meta: &vllm_models::AttentionMetadata,
        block_size: usize,
        device: &mut GpuDevice,
    ) -> ExecutorResult<(GpuTensor, GpuTensor, GpuTensor, GpuTensor, usize, usize)> {
        let num_reqs = meta.num_reqs;

        // cu_seqlens_q: cumulative query lengths [num_reqs + 1].
        let cu_seqlens_q: Vec<u32> = meta.query_start_loc.iter().map(|&x| x as u32).collect();
        let gpu_cu_seqlens_q = Self::h2d_u32(&cu_seqlens_q, device)?;

        // cu_seqlens_k: cumulative key lengths [num_reqs + 1].
        let mut cu_seqlens_k = Vec::with_capacity(num_reqs + 1);
        cu_seqlens_k.push(0u32);
        let mut cum = 0u32;
        for &sl in &meta.seq_lens {
            cum += sl as u32;
            cu_seqlens_k.push(cum);
        }
        let gpu_cu_seqlens_k = Self::h2d_u32(&cu_seqlens_k, device)?;

        let max_seqlen_q = meta.q_lens.iter().copied().max().unwrap_or(0);
        let max_seqlen_k = meta.seq_lens.iter().copied().max().unwrap_or(0);

        // slot_mapping: for each new token, compute (block_id * block_size + offset).
        let mut slot_mapping = Vec::with_capacity(meta.total_tokens);
        for i in 0..num_reqs {
            let tokens_before = meta.tokens_before[i];
            let q_len = meta.q_lens[i];
            let block_ids = &meta.block_ids[i];
            for t in 0..q_len {
                let abs_pos = tokens_before + t;
                let block_idx = abs_pos / block_size;
                let offset = abs_pos % block_size;
                if block_idx < block_ids.len() {
                    slot_mapping.push((block_ids[block_idx] * block_size + offset) as i64);
                } else {
                    slot_mapping.push(-1i64);
                }
            }
        }
        let gpu_slot_mapping = Self::h2d_i64(&slot_mapping, device)?;

        // block_table: [num_reqs, max_blocks_per_seq] u32 (padded with 0).
        let max_blocks = meta.block_ids.iter().map(|b| b.len()).max().unwrap_or(0);
        let gpu_block_table = if max_blocks > 0 {
            let mut block_table = vec![0u32; num_reqs * max_blocks];
            for (i, blocks) in meta.block_ids.iter().enumerate() {
                for (j, &bid) in blocks.iter().enumerate() {
                    block_table[i * max_blocks + j] = bid as u32;
                }
            }
            let t = Self::h2d_u32(&block_table, device)?;
            GpuTensor::new(t.raw_ptr(), &[num_reqs, max_blocks], GpuDType::U32)
        } else {
            GpuTensor::new(std::ptr::null_mut(), &[0, 0], GpuDType::U32)
        };

        Ok((
            gpu_slot_mapping,
            gpu_cu_seqlens_q,
            gpu_cu_seqlens_k,
            gpu_block_table,
            max_seqlen_q,
            max_seqlen_k,
        ))
    }

    /// D2H copy logits to CPU f32 vec.
    unsafe fn logits_to_cpu(
        logits: GpuTensor,
        device: &GpuDevice,
    ) -> ExecutorResult<Vec<f32>> {
        let num_elements = logits.num_elements();
        let nbytes = num_elements * logits.dtype().size_bytes();

        let mut host_buf = vec![0u8; nbytes];
        driver::memcpy_dtoh_async(
            host_buf.as_mut_ptr(),
            logits.raw_ptr() as *const u8,
            nbytes,
            device.compute_stream,
        )
        .map_err(|e| ExecutorError::WorkerExecution(format!("D2H logits: {e}")))?;
        driver::stream_synchronize(device.compute_stream)
            .map_err(|e| ExecutorError::WorkerExecution(format!("sync: {e}")))?;

        let f32_vec = match logits.dtype() {
            GpuDType::F32 => {
                let ptr = host_buf.as_ptr() as *const f32;
                std::slice::from_raw_parts(ptr, num_elements).to_vec()
            }
            GpuDType::F16 => {
                let ptr = host_buf.as_ptr() as *const half::f16;
                let slice = std::slice::from_raw_parts(ptr, num_elements);
                slice.iter().map(|v| v.to_f32()).collect()
            }
            GpuDType::BF16 => {
                let ptr = host_buf.as_ptr() as *const half::bf16;
                let slice = std::slice::from_raw_parts(ptr, num_elements);
                slice.iter().map(|v| v.to_f32()).collect()
            }
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "unexpected logits dtype".into(),
                ))
            }
        };
        Ok(f32_vec)
    }
}

// ---------------------------------------------------------------------------
// Worker trait implementation
// ---------------------------------------------------------------------------

impl Worker for CudaWorker {
    fn init_device(&mut self) -> ExecutorResult<()> {
        let device = GpuDevice::new(self.config.device_id)
            .map_err(|e| ExecutorError::WorkerInit(format!("GpuDevice init failed: {e}")))?;
        self.device = Some(device);
        Ok(())
    }

    fn load_model(&mut self) -> ExecutorResult<()> {
        let t0 = std::time::Instant::now();
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("device not initialized".into()))?;

        // 1. Resolve model directory.
        let model_dir = self.resolve_model_path()?;
        info!("CudaWorker: loading model from {}", model_dir.display());

        // Tokenizer on background thread.
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
            .map_err(|e| ExecutorError::WorkerInit(format!("config.json parse failed: {e}")))?;

        // 3. Resolve dtype.
        let dtype = match self.config.dtype.as_str() {
            "f16" | "float16" => GpuDType::F16,
            "bf16" | "bfloat16" => GpuDType::BF16,
            "f32" | "float32" => GpuDType::F32,
            _ => match hf_config.torch_dtype.as_deref() {
                Some("bfloat16") => GpuDType::BF16,
                Some("float16") => GpuDType::F16,
                _ => GpuDType::BF16,
            },
        };
        info!("CudaWorker: using dtype {:?}", dtype);

        // 4. Look up architecture.
        let arch = hf_config.architectures.first().cloned().unwrap_or_default();
        info!("CudaWorker: architecture = {arch}");

        // 5. Load weights into GPU memory.
        let mut weights = GpuWeights::from_dir(&model_dir, device.compute_stream)
            .map_err(|e| ExecutorError::WorkerInit(format!("weight load failed: {e}")))?;
        info!(
            "CudaWorker: loaded {} weight tensors",
            weights.len()
        );

        // 6. Construct model based on architecture.
        let model = match arch.as_str() {
            "LlamaForCausalLM" | "MistralForCausalLM" => {
                let config = llama_config_from_hf(&hf_config)?;
                let m = vllm_cuda::model::llama::LlamaForCausalLM::load(
                    &mut weights, &config, dtype, device,
                )
                .map_err(|e| ExecutorError::WorkerInit(format!("LlamaForCausalLM load: {e}")))?;
                CudaModel::Llama(m)
            }
            "Qwen2ForCausalLM" | "Qwen2_5ForCausalLM" => {
                let llama_config = llama_config_from_hf(&hf_config)?;
                let qwen2_config =
                    vllm_cuda::model::qwen2::Qwen2Config::from_llama_config(llama_config);
                let m = vllm_cuda::model::qwen2::Qwen2ForCausalLM::load(
                    &mut weights, &qwen2_config, dtype, device,
                )
                .map_err(|e| ExecutorError::WorkerInit(format!("Qwen2 load: {e}")))?;
                CudaModel::Qwen2(m)
            }
            "Gemma2ForCausalLM" => {
                let config = gemma2_config_from_hf(&hf_config)?;
                let m = vllm_cuda::model::gemma2::Gemma2ForCausalLM::load(
                    &mut weights, &config, dtype, device,
                )
                .map_err(|e| ExecutorError::WorkerInit(format!("Gemma2 load: {e}")))?;
                CudaModel::Gemma2(m)
            }
            _ => {
                return Err(ExecutorError::WorkerInit(format!(
                    "unsupported architecture for cuda-backend: {arch}. \
                     Supported: LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM"
                )));
            }
        };

        self.model_dtype = dtype;
        self.resolved_architecture = Some(arch);
        self.model = Some(model);
        self.model_dir = Some(model_dir);
        self.hf_config = Some(hf_config);

        // Collect tokenizer.
        if let Ok(Some(tok)) = tokenizer_handle.join() {
            self.preloaded_tokenizer = Some(tok);
        }

        info!(
            "CudaWorker: model loaded in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        Ok(())
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        _num_cpu_blocks: usize,
    ) -> ExecutorResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("model not loaded".into()))?;

        let pool = unsafe {
            KvCachePool::new(
                model.num_layers(),
                num_gpu_blocks,
                self.config.block_size,
                model.num_kv_heads(),
                model.head_dim(),
                self.model_dtype,
            )
        }
        .map_err(|e| ExecutorError::WorkerInit(format!("KvCachePool: {e}")))?;

        self.kv_cache = Some(pool);
        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        let (free, _total) = cudarc::driver::result::mem_get_info()
            .map_err(|e| ExecutorError::WorkerInit(format!("cuMemGetInfo: {e}")))?;
        info!(
            "CudaWorker: {:.0} MB free VRAM",
            free as f64 / 1_048_576.0
        );
        Ok(free)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        let block_size = self.config.block_size;

        // Clean up finished requests.
        for req_id in &scheduler_output.finished_req_ids {
            self.token_buffers.remove(req_id);
            self.sampling_params_map.remove(req_id);
        }
        self.input_batch
            .remove_finished(&scheduler_output.finished_req_ids);

        // Process newly scheduled requests.
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
            let start = new_req.num_computed_tokens as usize;
            let end = (start + num_tokens).min(prompt_ids.len());
            let tokens_to_use = &prompt_ids[start..end];

            self.token_buffers
                .insert(new_req.req_id.clone(), prompt_ids.to_vec());
            if let Some(ref params) = new_req.sampling_params {
                self.sampling_params_map
                    .insert(new_req.req_id.clone(), params.clone());
            }

            let block_ids = new_req.block_ids.first().cloned().unwrap_or_default();
            self.input_batch.add_request(
                new_req.req_id.clone(),
                tokens_to_use,
                block_ids,
                new_req.num_computed_tokens,
            );
        }

        // Update cached requests' block tables.
        for (i, req_id) in scheduler_output
            .scheduled_cached_reqs
            .req_ids
            .iter()
            .enumerate()
        {
            if let Some(Some(new_blocks)) =
                scheduler_output.scheduled_cached_reqs.new_block_ids.get(i)
                && let Some(group0) = new_blocks.first()
            {
                self.input_batch.update_blocks(req_id, group0.clone());
            }
        }

        // Prepare flat inputs from InputBatch.
        let prepared = self
            .input_batch
            .prepare_inputs(&scheduler_output.scheduled_spec_decode_tokens);
        if prepared.flat_token_ids.is_empty() {
            return Ok(ModelRunnerOutput::from_token_map(HashMap::new()));
        }

        // Split borrows: model + kv_cache (shared) vs device (mutable).
        // Use direct field access so the borrow checker sees disjoint borrows.
        let (model, kv_cache, device) = match (&self.model, &self.kv_cache, &mut self.device) {
            (Some(m), Some(kv), Some(d)) => (m, kv, d),
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "model, KV cache, or device not initialized".into(),
                ));
            }
        };
        let vocab_size = model.vocab_size();

        // Reset scratch arena for this step.
        device.arena.reset();

        // H2D: input_ids and positions.
        let (gpu_input_ids, gpu_positions) = unsafe {
            let ids = Self::h2d_u32(&prepared.flat_token_ids, device)?;
            let pos = Self::h2d_u32(&prepared.flat_positions, device)?;
            (ids, pos)
        };

        // Build attention metadata tensors.
        let (slot_mapping, cu_seqlens_q, cu_seqlens_k, block_table, max_seqlen_q, max_seqlen_k) =
            unsafe { Self::build_attention_tensors(&prepared.attn_meta, block_size, device)? };

        // Forward pass.
        let logits = unsafe {
            model.forward(
                gpu_input_ids,
                gpu_positions,
                slot_mapping,
                cu_seqlens_q,
                cu_seqlens_k,
                block_table,
                max_seqlen_q,
                max_seqlen_k,
                kv_cache,
                device,
            )
        };

        // D2H logits for CPU sampling.
        let logits_f32 = unsafe { Self::logits_to_cpu(logits, device)? };

        // Per-request sampling.
        let mut sampler = Sampler::new();
        let mut token_map: HashMap<String, Vec<u32>> = HashMap::new();

        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
            let start = req_idx * vocab_size;
            let end = start + vocab_size;
            let req_logits = &logits_f32[start..end];

            let prev_tokens = self
                .token_buffers
                .get(&req_slice.req_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            let sampled =
                if let Some(params) = self.sampling_params_map.get(&req_slice.req_id) {
                    let (token_id, _logprobs) =
                        sampler.sample_one(req_logits, params, prev_tokens, None);
                    vec![token_id]
                } else {
                    // Greedy argmax.
                    let token_id = req_logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0);
                    vec![token_id]
                };

            self.input_batch.commit_step(
                &req_slice.req_id,
                &sampled,
                req_slice.token_count,
                !req_slice.spec_token_ids.is_empty(),
            );
            if let Some(buf) = self.token_buffers.get_mut(&req_slice.req_id) {
                buf.extend_from_slice(&sampled);
            }
            token_map.insert(req_slice.req_id.clone(), sampled);
        }

        Ok(ModelRunnerOutput::from_token_map(token_map))
    }

    fn compile_or_warm_up_model(&mut self) -> ExecutorResult<()> {
        // TODO: CUDA graph capture for decode batches.
        Ok(())
    }

    fn shutdown(&mut self) {
        self.model = None;
        self.kv_cache = None;
        self.device = None;
        self.is_shutdown = true;
    }

    fn rank(&self) -> usize {
        0
    }

    fn local_rank(&self) -> usize {
        self.config.device_id as usize
    }

    fn is_driver_worker(&self) -> bool {
        true
    }

    fn take_preloaded_tokenizer(&mut self) -> Option<tokenizers::Tokenizer> {
        self.preloaded_tokenizer.take()
    }

    fn architecture(&self) -> Option<String> {
        self.resolved_architecture.clone()
    }
}
