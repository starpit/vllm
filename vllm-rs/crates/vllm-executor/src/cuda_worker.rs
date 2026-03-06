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

use tracing::info;
use vllm_common::SamplingParams;
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_cuda::device::GpuDevice;
use vllm_cuda::driver;
use vllm_cuda::dtype::DType as GpuDType;
use vllm_cuda::graph::{CudaGraphRunner, GRAPH_MAX_BLOCKS_PER_SEQ};
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
    /// Skip CUDA graph capture (--enforce-eager).
    pub enforce_eager: bool,
    /// Maximum tokens per scheduler iteration (controls arena pre-sizing).
    pub max_num_batched_tokens: usize,
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
        last_token_indices: Option<GpuTensor>,
    ) -> GpuTensor {
        match self {
            Self::Llama(m) => unsafe {
                m.forward(
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                    last_token_indices,
                )
            },
            Self::Qwen2(m) => unsafe {
                m.forward(
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                    last_token_indices,
                )
            },
            Self::Gemma2(m) => unsafe {
                m.forward(
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                    last_token_indices,
                )
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Config helpers: HfModelConfig → vllm-cuda configs
// ---------------------------------------------------------------------------

fn llama_config_from_hf(
    hf: &HfModelConfig,
) -> ExecutorResult<vllm_cuda::model::llama::LlamaConfig> {
    let hidden_size = hf
        .hidden_size
        .ok_or_else(|| ExecutorError::WorkerInit("missing hidden_size in config.json".into()))?;
    let num_attention_heads = hf
        .num_attention_heads
        .ok_or_else(|| ExecutorError::WorkerInit("missing num_attention_heads".into()))?;
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

fn gemma2_config_from_hf(
    hf: &HfModelConfig,
) -> ExecutorResult<vllm_cuda::model::gemma2::Gemma2Config> {
    let hidden_size = hf
        .hidden_size
        .ok_or_else(|| ExecutorError::WorkerInit("missing hidden_size".into()))?;
    let num_attention_heads = hf
        .num_attention_heads
        .ok_or_else(|| ExecutorError::WorkerInit("missing num_attention_heads".into()))?;
    let num_kv_heads = hf.num_key_value_heads.unwrap_or(num_attention_heads);
    let num_hidden_layers = hf.num_hidden_layers.unwrap_or(26);
    let head_dim = hf.head_dim.unwrap_or(hidden_size / num_attention_heads);

    let query_pre_attn_scalar = hf
        .extra
        .get("query_pre_attn_scalar")
        .and_then(|v| v.as_f64())
        .unwrap_or(head_dim as f64);
    let attn_logit_softcapping = hf
        .extra
        .get("attn_logit_softcapping")
        .and_then(|v| v.as_f64());
    let final_logit_softcapping = hf
        .extra
        .get("final_logit_softcapping")
        .and_then(|v| v.as_f64());
    let sliding_window = hf
        .extra
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    // Gemma2 alternates sliding/full attention: even layers = sliding, odd = full.
    // Or it may be specified via sliding_window_pattern in config.
    let layer_is_sliding: Vec<bool> = (0..num_hidden_layers).map(|i| i % 2 == 0).collect();

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
    /// Backing GPU buffers for weight tensors. Must outlive `model`.
    _weights: Option<GpuWeights>,
    /// CUDA graph runner for decode batches.
    graph_runner: Option<CudaGraphRunner>,
    /// Batch size of the last graph replay. When the batch composition is
    /// unchanged, we can skip the input_ids H2D copy because the graph's
    /// D2D scatter already placed argmax results into the persistent buffer.
    last_graph_batch_size: Option<usize>,
    /// True when persistent graph buffers contain valid metadata from a
    /// previous replay (positions, slot_mapping, cu_seqlens_k). When true,
    /// we use `replay_decode_fast()` which updates metadata on GPU instead
    /// of building Vecs on CPU and doing H2D copies.
    graph_metadata_valid: bool,

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
            _weights: None,
            graph_runner: None,
            last_graph_batch_size: None,
            graph_metadata_valid: false,
            token_buffers: HashMap::new(),
            sampling_params_map: HashMap::new(),
            input_batch: InputBatch::new(),
            preloaded_tokenizer: None,
        }
    }

    /// Update max_num_batched_tokens (used for GPU-aware auto-detection).
    pub fn set_max_num_batched_tokens(&mut self, value: usize) {
        self.config.max_num_batched_tokens = value;
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
        let api = builder
            .build()
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to build HF API: {e}")))?;
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
    fn h2d_u32(data: &[u32], device: &mut GpuDevice) -> ExecutorResult<GpuTensor> {
        let t = device.arena.alloc(&[data.len()], GpuDType::U32);
        unsafe {
            driver::memcpy_htod_async(
                t.raw_ptr(),
                data.as_ptr() as *const u8,
                data.len() * 4,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("H2D u32: {e}")))?;
        Ok(t)
    }

    /// H2D copy an i64 slice into an arena-allocated GpuTensor.
    fn h2d_i64(data: &[i64], device: &mut GpuDevice) -> ExecutorResult<GpuTensor> {
        let t = device.arena.alloc(&[data.len()], GpuDType::I64);
        unsafe {
            driver::memcpy_htod_async(
                t.raw_ptr(),
                data.as_ptr() as *const u8,
                data.len() * 8,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("H2D i64: {e}")))?;
        Ok(t)
    }

    /// Build attention metadata tensors from `AttentionMetadata`.
    ///
    /// Returns `(slot_mapping, cu_seqlens_q, cu_seqlens_k, block_table, max_seqlen_q, max_seqlen_k)`.
    fn build_attention_tensors(
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
            unsafe { GpuTensor::new(t.raw_ptr(), &[num_reqs, max_blocks], GpuDType::U32) }
        } else {
            unsafe { GpuTensor::new(std::ptr::null_mut(), &[0, 0], GpuDType::U32) }
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
    fn logits_to_cpu(logits: GpuTensor, device: &GpuDevice) -> ExecutorResult<Vec<f32>> {
        let num_elements = logits.numel();
        let nbytes = num_elements * logits.dtype().size_bytes();

        let mut host_buf = vec![0u8; nbytes];
        unsafe {
            driver::memcpy_dtoh_async(
                host_buf.as_mut_ptr(),
                logits.raw_ptr() as *const u8,
                nbytes,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("D2H logits: {e}")))?;
        unsafe { driver::stream_synchronize(device.compute_stream) }
            .map_err(|e| ExecutorError::WorkerExecution(format!("sync: {e}")))?;

        let f32_vec: Vec<f32> = match logits.dtype() {
            GpuDType::F32 => {
                let ptr = host_buf.as_ptr() as *const f32;
                unsafe { std::slice::from_raw_parts(ptr, num_elements) }.to_vec()
            }
            GpuDType::F16 => {
                let ptr = host_buf.as_ptr() as *const half::f16;
                let slice = unsafe { std::slice::from_raw_parts(ptr, num_elements) };
                slice.iter().map(|v| v.to_f32()).collect()
            }
            GpuDType::BF16 => {
                let ptr = host_buf.as_ptr() as *const half::bf16;
                let slice = unsafe { std::slice::from_raw_parts(ptr, num_elements) };
                slice.iter().map(|v| v.to_f32()).collect()
            }
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "unexpected logits dtype".into(),
                ));
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

        // Ensure CUDA context is current on this thread (may differ from init_device thread).
        unsafe { driver::ctx_set_current(device.ctx) }
            .map_err(|e| ExecutorError::WorkerInit(format!("ctx_set_current: {e}")))?;

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
        // Sync to ensure all H2D weight copies are complete before D2D concat.
        unsafe { driver::stream_synchronize(device.compute_stream) }
            .map_err(|e| ExecutorError::WorkerInit(format!("weight sync: {e}")))?;
        info!("CudaWorker: loaded {} weight tensors", weights.len());

        // 6. Construct model based on architecture.
        let model = match arch.as_str() {
            "LlamaForCausalLM" | "MistralForCausalLM" => {
                let config = llama_config_from_hf(&hf_config)?;
                let m = vllm_cuda::model::llama::LlamaForCausalLM::load(
                    &mut weights,
                    &config,
                    dtype,
                    device,
                )
                .map_err(|e| ExecutorError::WorkerInit(format!("LlamaForCausalLM load: {e}")))?;
                CudaModel::Llama(m)
            }
            "Qwen2ForCausalLM" | "Qwen2_5ForCausalLM" => {
                let llama_config = llama_config_from_hf(&hf_config)?;
                let qwen2_config =
                    vllm_cuda::model::qwen2::Qwen2Config::from_llama_config(llama_config);
                let m = vllm_cuda::model::qwen2::Qwen2ForCausalLM::load(
                    &mut weights,
                    &qwen2_config,
                    dtype,
                    device,
                )
                .map_err(|e| ExecutorError::WorkerInit(format!("Qwen2 load: {e}")))?;
                CudaModel::Qwen2(m)
            }
            "Gemma2ForCausalLM" => {
                let config = gemma2_config_from_hf(&hf_config)?;
                let m = vllm_cuda::model::gemma2::Gemma2ForCausalLM::load(
                    &mut weights,
                    &config,
                    dtype,
                    device,
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
        self._weights = Some(weights); // keep backing GPU buffers alive
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
        if let Some(ref dev) = self.device {
            unsafe { driver::ctx_set_current(dev.ctx) }
                .map_err(|e| ExecutorError::WorkerInit(format!("ctx_set_current: {e}")))?;
        }
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
        if let Some(ref dev) = self.device {
            unsafe { driver::ctx_set_current(dev.ctx) }
                .map_err(|e| ExecutorError::WorkerInit(format!("ctx_set_current: {e}")))?;
        }
        let (free, _total) = cudarc::driver::result::mem_get_info()
            .map_err(|e| ExecutorError::WorkerInit(format!("cuMemGetInfo: {e}")))?;
        info!("CudaWorker: {:.0} MB free VRAM", free as f64 / 1_048_576.0);
        Ok(free)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        // Ensure CUDA context is current on this thread. In async scheduling,
        // execute_model runs on a dedicated executor thread that differs from
        // the init thread where the context was created.
        if let Some(ref dev) = self.device {
            unsafe { driver::ctx_set_current(dev.ctx) }
                .map_err(|e| ExecutorError::WorkerExecution(format!("ctx_set_current: {e}")))?;
        }

        let block_size = self.config.block_size;

        // Clean up finished requests.
        if !scheduler_output.finished_req_ids.is_empty()
            || !scheduler_output.scheduled_new_reqs.is_empty()
        {
            // Batch composition changed — can't reuse persistent input_ids or metadata.
            self.last_graph_batch_size = None;
            self.graph_metadata_valid = false;
        }
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

        // Update cached requests' block tables. Track if any changed.
        let mut blocks_changed = !scheduler_output.scheduled_new_reqs.is_empty();
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
                if !group0.is_empty() {
                    blocks_changed = true;
                }
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
        let num_reqs = prepared.req_inputs.len();
        let total_tokens = prepared.flat_token_ids.len();

        // Check if this is a pure decode batch (all q_len=1) and we have a graph.
        // We allow padding to the nearest captured graph size (e.g. BS=3 → graph BS=4).
        let is_decode = prepared.attn_meta.q_lens.iter().all(|&q| q == 1);
        let graph_bs = if is_decode {
            self.graph_runner
                .as_ref()
                .and_then(|r| r.nearest_graph_size(num_reqs))
        } else {
            None
        };
        let use_graph = graph_bs.is_some();

        // Check if all requests are greedy (temp < 1e-6). Used to decide
        // whether to use the in-graph argmax fast path.
        let all_greedy = prepared.req_inputs.iter().all(|r| {
            self.sampling_params_map
                .get(&r.req_id)
                .is_none_or(|p| p.temperature < 1e-6)
        });

        if use_graph && all_greedy {
            // Fast path: CUDA graph with in-graph argmax. No separate sampling
            // kernel launch — argmax + D2D scatter are captured in the graph.
            let graph_bs = graph_bs.unwrap();
            let meta = &prepared.attn_meta;

            let replay_out =
                if self.graph_metadata_valid && self.last_graph_batch_size == Some(graph_bs) {
                    // GPU-side metadata update: positions, slot_mapping, cu_seqlens_k
                    // are incremented on GPU in a single kernel. Only block_table is
                    // H2D-copied when blocks changed. input_ids were scattered by the
                    // previous graph replay's in-graph argmax.
                    let new_bt = if blocks_changed {
                        let mut block_table = vec![0u32; graph_bs * GRAPH_MAX_BLOCKS_PER_SEQ];
                        for (i, blocks) in meta.block_ids.iter().enumerate() {
                            for (j, &bid) in blocks.iter().enumerate() {
                                if j < GRAPH_MAX_BLOCKS_PER_SEQ {
                                    block_table[i * GRAPH_MAX_BLOCKS_PER_SEQ + j] = bid as u32;
                                }
                            }
                        }
                        Some(block_table)
                    } else {
                        None
                    };

                    let runner = self.graph_runner.as_ref().unwrap();
                    unsafe {
                        runner.replay_decode_fast(
                            graph_bs,
                            None, // input_ids already scattered by previous graph
                            new_bt.as_deref(),
                            block_size,
                            device,
                        )
                    }
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("graph replay_decode_fast: {e}"))
                    })?
                } else {
                    // First step for this batch or batch composition changed:
                    // full H2D of all metadata to initialize persistent buffers.
                    let mut input_ids = prepared.flat_token_ids.clone();
                    input_ids.resize(graph_bs, 0);
                    let mut positions = prepared.flat_positions.clone();
                    positions.resize(graph_bs, 0);

                    let mut cu_seqlens_q: Vec<u32> = (0..=num_reqs as u32).collect();
                    for _ in num_reqs..graph_bs {
                        cu_seqlens_q.push(num_reqs as u32);
                    }

                    let mut cu_seqlens_k = Vec::with_capacity(graph_bs + 1);
                    cu_seqlens_k.push(0u32);
                    let mut cum = 0u32;
                    for &sl in &meta.seq_lens {
                        cum += sl as u32;
                        cu_seqlens_k.push(cum);
                    }
                    for _ in num_reqs..graph_bs {
                        cu_seqlens_k.push(cum);
                    }

                    let mut slot_mapping = Vec::with_capacity(graph_bs);
                    for i in 0..num_reqs {
                        let abs_pos = meta.tokens_before[i];
                        let block_idx = abs_pos / block_size;
                        let offset = abs_pos % block_size;
                        let block_ids = &meta.block_ids[i];
                        if block_idx < block_ids.len() {
                            slot_mapping.push((block_ids[block_idx] * block_size + offset) as i64);
                        } else {
                            slot_mapping.push(-1i64);
                        }
                    }
                    slot_mapping.resize(graph_bs, -1i64);

                    let mut block_table = vec![0u32; graph_bs * GRAPH_MAX_BLOCKS_PER_SEQ];
                    for (i, blocks) in meta.block_ids.iter().enumerate() {
                        for (j, &bid) in blocks.iter().enumerate() {
                            if j < GRAPH_MAX_BLOCKS_PER_SEQ {
                                block_table[i * GRAPH_MAX_BLOCKS_PER_SEQ + j] = bid as u32;
                            }
                        }
                    }

                    let skip_input_ids =
                        self.last_graph_batch_size == Some(graph_bs) && num_reqs == graph_bs;

                    let runner = self.graph_runner.as_ref().unwrap();
                    unsafe {
                        runner.replay(
                            graph_bs,
                            &input_ids,
                            &positions,
                            &slot_mapping,
                            &cu_seqlens_q,
                            &cu_seqlens_k,
                            &block_table,
                            device,
                            skip_input_ids,
                        )
                    }
                    .map_err(|e| ExecutorError::WorkerExecution(format!("graph replay: {e}")))?
                };

            // D2H the argmax token IDs (only num_reqs × 4 bytes — skip padded slots).
            let mut host_ids = vec![0u32; num_reqs];
            unsafe {
                driver::memcpy_dtoh_async(
                    host_ids.as_mut_ptr() as *mut u8,
                    replay_out.token_ids.raw_ptr() as *const u8,
                    num_reqs * 4,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("D2H token ids: {e}")))?;
            unsafe { driver::stream_synchronize(device.compute_stream) }
                .map_err(|e| ExecutorError::WorkerExecution(format!("sync: {e}")))?;

            // Record that graph buffers now have valid metadata for next step.
            self.last_graph_batch_size = Some(graph_bs);
            self.graph_metadata_valid = true;

            let mut token_map: HashMap<String, Vec<u32>> = HashMap::new();
            for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
                let sampled = vec![host_ids[req_idx]];
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
            return Ok(ModelRunnerOutput::from_token_map(token_map));
        }

        // Non-greedy graph path or eager path: need separate sampling.
        let logits = if use_graph {
            // CUDA graph replay (non-greedy: in-graph argmax result is
            // discarded; we re-sample with temperature on the logits).
            let graph_bs = graph_bs.unwrap();
            let meta = &prepared.attn_meta;

            let mut input_ids = prepared.flat_token_ids.clone();
            input_ids.resize(graph_bs, 0);
            let mut positions = prepared.flat_positions.clone();
            positions.resize(graph_bs, 0);

            let mut cu_seqlens_q: Vec<u32> = (0..=num_reqs as u32).collect();
            for _ in num_reqs..graph_bs {
                cu_seqlens_q.push(num_reqs as u32);
            }

            let mut cu_seqlens_k = Vec::with_capacity(graph_bs + 1);
            cu_seqlens_k.push(0u32);
            let mut cum = 0u32;
            for &sl in &meta.seq_lens {
                cum += sl as u32;
                cu_seqlens_k.push(cum);
            }
            for _ in num_reqs..graph_bs {
                cu_seqlens_k.push(cum);
            }

            let mut slot_mapping = Vec::with_capacity(graph_bs);
            for i in 0..num_reqs {
                let abs_pos = meta.tokens_before[i];
                let block_idx = abs_pos / block_size;
                let offset = abs_pos % block_size;
                let block_ids = &meta.block_ids[i];
                if block_idx < block_ids.len() {
                    slot_mapping.push((block_ids[block_idx] * block_size + offset) as i64);
                } else {
                    slot_mapping.push(-1i64);
                }
            }
            slot_mapping.resize(graph_bs, -1i64);

            let mut block_table = vec![0u32; graph_bs * GRAPH_MAX_BLOCKS_PER_SEQ];
            for (i, blocks) in meta.block_ids.iter().enumerate() {
                for (j, &bid) in blocks.iter().enumerate() {
                    if j < GRAPH_MAX_BLOCKS_PER_SEQ {
                        block_table[i * GRAPH_MAX_BLOCKS_PER_SEQ + j] = bid as u32;
                    }
                }
            }

            // Non-greedy: don't skip input_ids and don't track for next step.
            self.last_graph_batch_size = None;
            self.graph_metadata_valid = false;

            let runner = self.graph_runner.as_ref().unwrap();
            let replay_out = unsafe {
                runner.replay(
                    graph_bs,
                    &input_ids,
                    &positions,
                    &slot_mapping,
                    &cu_seqlens_q,
                    &cu_seqlens_k,
                    &block_table,
                    device,
                    false, // always H2D input_ids for non-greedy
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("graph replay: {e}")))?;
            // Slice logits to only the real requests (discard padded rows).
            if graph_bs > num_reqs {
                replay_out.logits.narrow_dim0(0, num_reqs)
            } else {
                replay_out.logits
            }
        } else {
            // Eager forward path (prefill or uncaptured batch size).
            self.last_graph_batch_size = None;
            self.graph_metadata_valid = false;
            device.arena.reset();

            let gpu_input_ids = Self::h2d_u32(&prepared.flat_token_ids, device)?;
            let gpu_positions = Self::h2d_u32(&prepared.flat_positions, device)?;

            let (slot_mapping, cu_seqlens_q, cu_seqlens_k, block_table, max_seqlen_q, max_seqlen_k) =
                Self::build_attention_tensors(&prepared.attn_meta, block_size, device)?;

            let last_token_indices = if num_reqs < total_tokens {
                let mut indices = Vec::with_capacity(num_reqs);
                let mut offset = 0u32;
                for req_slice in &prepared.req_inputs {
                    indices.push(offset + req_slice.token_count as u32 - 1);
                    offset += req_slice.token_count as u32;
                }
                Some(Self::h2d_u32(&indices, device)?)
            } else {
                None
            };

            unsafe {
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
                    last_token_indices,
                )
            }
        };

        let mut token_map: HashMap<String, Vec<u32>> = HashMap::new();

        // Check if all requests can use GPU sampling (no grammar, no logit_bias,
        // no frequency/presence/repetition penalties). This covers both greedy
        // (temp < 1e-6 → argmax) and non-greedy (temp >= 1e-6 → top-k/top-p/min-p).
        let all_gpu_sampleable = prepared.req_inputs.iter().all(|r| {
            self.sampling_params_map.get(&r.req_id).is_none_or(|p| {
                p.frequency_penalty == 0.0
                    && p.presence_penalty == 0.0
                    && p.repetition_penalty == 1.0
                    && p.logit_bias.is_none()
            })
        });

        if all_gpu_sampleable && !all_greedy {
            // GPU sampling fast path: batched top-k/top-p/min-p on device.
            // Pack all 5 param arrays into a single contiguous H2D copy to
            // minimize driver overhead (1 copy instead of 5).
            use rand::Rng;
            let mut rng = rand::thread_rng();

            // Layout: [temps(N×f32), top_ks(N×i32), top_ps(N×f32), min_ps(N×f32), randoms(N×f32)]
            let stride = num_reqs * 4; // bytes per array
            let total_bytes = stride * 5;
            let mut packed = vec![0u8; total_bytes];

            let temps_ptr = packed.as_mut_ptr() as *mut f32;
            let top_ks_ptr = unsafe { packed.as_mut_ptr().add(stride) as *mut i32 };
            let top_ps_ptr = unsafe { packed.as_mut_ptr().add(stride * 2) as *mut f32 };
            let min_ps_ptr = unsafe { packed.as_mut_ptr().add(stride * 3) as *mut f32 };
            let randoms_ptr = unsafe { packed.as_mut_ptr().add(stride * 4) as *mut f32 };

            for (i, req_slice) in prepared.req_inputs.iter().enumerate() {
                let params = self.sampling_params_map.get(&req_slice.req_id);
                let (t, k, p, mp) = params.map_or((1.0f32, 0i32, 1.0f32, 0.0f32), |p| {
                    (
                        p.temperature.max(1e-7) as f32,
                        p.top_k,
                        p.top_p as f32,
                        p.min_p as f32,
                    )
                });
                unsafe {
                    *temps_ptr.add(i) = t;
                    *top_ks_ptr.add(i) = k;
                    *top_ps_ptr.add(i) = p;
                    *min_ps_ptr.add(i) = mp;
                    *randoms_ptr.add(i) = rng.r#gen::<f32>();
                }
            }

            // Single H2D copy for all sampling params.
            let gpu_packed = device.arena.alloc(&[total_bytes / 4], GpuDType::F32);
            unsafe {
                driver::memcpy_htod_async(
                    gpu_packed.raw_ptr(),
                    packed.as_ptr(),
                    total_bytes,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("H2D sampling params: {e}")))?;

            // Create views into the packed GPU buffer.
            let base = gpu_packed.raw_ptr();
            let gpu_temps = unsafe { GpuTensor::new(base, &[num_reqs], GpuDType::F32) };
            let gpu_top_ks =
                unsafe { GpuTensor::new(base.add(stride), &[num_reqs], GpuDType::U32) };
            let gpu_top_ps =
                unsafe { GpuTensor::new(base.add(stride * 2), &[num_reqs], GpuDType::F32) };
            let gpu_min_ps =
                unsafe { GpuTensor::new(base.add(stride * 3), &[num_reqs], GpuDType::F32) };
            let gpu_randoms =
                unsafe { GpuTensor::new(base.add(stride * 4), &[num_reqs], GpuDType::F32) };

            let token_ids_gpu = unsafe {
                vllm_cuda::kernels::sample_batched(
                    logits,
                    gpu_temps,
                    gpu_top_ks,
                    gpu_top_ps,
                    gpu_min_ps,
                    gpu_randoms,
                    &mut device.arena,
                    device.compute_stream,
                )
            };

            let mut host_ids = vec![0u32; num_reqs];
            unsafe {
                driver::memcpy_dtoh_async(
                    host_ids.as_mut_ptr() as *mut u8,
                    token_ids_gpu.raw_ptr() as *const u8,
                    num_reqs * 4,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("D2H token ids: {e}")))?;
            unsafe { driver::stream_synchronize(device.compute_stream) }
                .map_err(|e| ExecutorError::WorkerExecution(format!("sync: {e}")))?;

            for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
                let sampled = vec![host_ids[req_idx]];
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
        } else if all_greedy {
            // GPU fast path: batched argmax on device, D2H only num_reqs × 4 bytes.
            let token_ids_gpu = unsafe {
                vllm_cuda::kernels::argmax_batched(logits, &mut device.arena, device.compute_stream)
            };
            let num_reqs = prepared.req_inputs.len();
            let mut host_ids = vec![0u32; num_reqs];
            unsafe {
                driver::memcpy_dtoh_async(
                    host_ids.as_mut_ptr() as *mut u8,
                    token_ids_gpu.raw_ptr() as *const u8,
                    num_reqs * 4,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("D2H token ids: {e}")))?;
            unsafe { driver::stream_synchronize(device.compute_stream) }
                .map_err(|e| ExecutorError::WorkerExecution(format!("sync: {e}")))?;

            for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
                let sampled = vec![host_ids[req_idx]];
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
        } else {
            // Slow path: D2H full logits for CPU sampling.
            let logits_f32 = Self::logits_to_cpu(logits, device)?;
            let mut sampler = Sampler::new();

            for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
                let start = req_idx * vocab_size;
                let end = start + vocab_size;
                let req_logits = &logits_f32[start..end];

                let prev_tokens = self
                    .token_buffers
                    .get(&req_slice.req_id)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);

                let sampled = if let Some(params) = self.sampling_params_map.get(&req_slice.req_id)
                {
                    let (token_id, _logprobs) =
                        sampler.sample_one(req_logits, params, prev_tokens, None);
                    vec![token_id]
                } else {
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
        }

        Ok(ModelRunnerOutput::from_token_map(token_map))
    }

    fn compile_or_warm_up_model(&mut self) -> ExecutorResult<()> {
        if self.config.enforce_eager {
            info!("CudaWorker: --enforce-eager set, skipping CUDA graph capture");
            return Ok(());
        }

        let (model, kv_cache, device) = match (&self.model, &self.kv_cache, &mut self.device) {
            (Some(m), Some(kv), Some(d)) => (m, kv, d),
            _ => return Ok(()), // Not fully initialized yet.
        };

        unsafe { driver::ctx_set_current(device.ctx) }
            .map_err(|e| ExecutorError::WorkerInit(format!("ctx_set_current: {e}")))?;

        let vocab_size = model.vocab_size();

        // Capture CUDA graphs for common decode batch sizes.
        // During decode, every request has q_len=1, so shapes are deterministic.
        let capture_sizes: Vec<usize> = vec![1, 2, 4, 8, 16, 32];
        let max_bs = *capture_sizes.iter().max().unwrap();

        // Pre-size the arena by running dummy forwards that cover the worst-case
        // allocation pattern. This ensures the arena won't grow after graph capture
        // (which would invalidate captured pointers since the arena base changes).
        // Run with a prefill-like token count (max_bs * 32) to cover both prefill
        // and decode arena needs.
        {
            // Pre-size the arena to handle the worst-case prefill batch, which is
            // max_num_batched_tokens tokens in a single forward pass.
            let prefill_tokens = self.config.max_num_batched_tokens;
            info!("Pre-sizing arena with dummy forward ({prefill_tokens} tokens)...");
            device.arena.reset();
            let dummy_ids = device.arena.alloc(&[prefill_tokens], GpuDType::U32);
            let dummy_pos = device.arena.alloc(&[prefill_tokens], GpuDType::U32);
            let dummy_slots = device.arena.alloc(&[prefill_tokens], GpuDType::I64);
            // Zero slot_mapping and block_table to avoid out-of-bounds KV cache writes.
            unsafe {
                driver::memset_d8(
                    dummy_slots.raw_ptr(),
                    0,
                    prefill_tokens * 8,
                    device.compute_stream,
                )
                .ok();
            }
            // cu_seqlens: single sequence of prefill_tokens length (worst case for arena).
            let cu_q: Vec<u32> = vec![0, prefill_tokens as u32];
            let gpu_cu_q = device.arena.alloc(&[2], GpuDType::U32);
            unsafe {
                driver::memcpy_htod_async(
                    gpu_cu_q.raw_ptr(),
                    cu_q.as_ptr() as *const u8,
                    2 * 4,
                    device.compute_stream,
                )
                .ok();
            }
            let gpu_cu_k = device.arena.alloc(&[2], GpuDType::U32);
            unsafe {
                driver::memcpy_htod_async(
                    gpu_cu_k.raw_ptr(),
                    cu_q.as_ptr() as *const u8,
                    2 * 4,
                    device.compute_stream,
                )
                .ok();
            }
            let dummy_bt = device.arena.alloc(&[1, 1], GpuDType::U32);
            unsafe {
                driver::memset_d8(dummy_bt.raw_ptr(), 0, 4, device.compute_stream).ok();
            }
            // max_seqlen_q = max_seqlen_k = prefill_tokens for arena sizing.
            unsafe {
                let _ = model.forward(
                    dummy_ids,
                    dummy_pos,
                    dummy_slots,
                    gpu_cu_q,
                    gpu_cu_k,
                    dummy_bt,
                    prefill_tokens,
                    prefill_tokens,
                    kv_cache,
                    device,
                    None,
                );
                let _ = driver::stream_synchronize(device.compute_stream);
            }
            device.arena.reset();
            info!(
                "Arena pre-sized: capacity={:.1} MB, high_water={:.1} MB",
                device.arena.capacity() as f64 / (1024.0 * 1024.0),
                device.arena.high_water_mark() as f64 / (1024.0 * 1024.0),
            );
        }

        let mut runner = unsafe { CudaGraphRunner::new(max_bs, vocab_size, self.model_dtype) }
            .map_err(|e| ExecutorError::WorkerInit(format!("CudaGraphRunner::new: {e}")))?;

        for &bs in &capture_sizes {
            info!("Capturing CUDA graph for batch_size={bs}...");
            let kv_ref = kv_cache;
            let model_ref = model;

            // Capture with max_seqlen_k padded to 2048. Paged FA2 uses per-sequence
            // lengths from cu_seqlens_k and iterates via block_table, so a large
            // max_seqlen_k just over-allocates workspace — correctness is maintained.
            // This avoids re-capture as sequences grow during inference.
            let padded_max_seqlen_k: usize = 2048;

            let result = unsafe {
                runner.capture(bs, device, |inputs, dev| {
                    model_ref.forward(
                        inputs.input_ids,
                        inputs.positions,
                        inputs.slot_mapping,
                        inputs.cu_seqlens_q,
                        inputs.cu_seqlens_k,
                        inputs.block_table,
                        1, // max_seqlen_q = 1 for decode
                        padded_max_seqlen_k,
                        kv_ref,
                        dev,
                        None, // no last_token_indices (decode: all tokens are last)
                    )
                })
            };

            match result {
                Ok(()) => info!("CUDA graph captured for batch_size={bs}"),
                Err(e) => {
                    tracing::warn!("Failed to capture CUDA graph for bs={bs}: {e}");
                    // Continue without graphs for this size.
                }
            }
        }

        if !runner.captured_sizes().is_empty() {
            info!(
                "CUDA graphs captured for batch sizes: {:?}",
                runner.captured_sizes()
            );
            self.graph_runner = Some(runner);
        }

        // Benchmark cublasLt algorithms now that the plan cache is populated
        // from both the prefill warmup and graph capture (decode shapes).
        // This replaces heuristic-selected algorithms with empirically fastest ones.
        unsafe { device.cublas.benchmark_plans(&mut device.arena) };
        device.arena.reset();

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
