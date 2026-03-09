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
use vllm_cuda::cpu_gpu_buf::PinnedBuf;
use vllm_cuda::device::GpuDevice;
use vllm_cuda::driver;
use vllm_cuda::dtype::DType as GpuDType;
use vllm_cuda::graph::{CudaGraphRunner, GRAPH_MAX_BLOCKS_PER_SEQ, PrefillGraphRunner};
use vllm_cuda::kv_cache::KvCachePool;
use vllm_cuda::tensor::GpuTensor;
use vllm_cuda::weights::GpuWeights;
use vllm_engine::executor::ModelRunnerOutput;
use vllm_model::weight::HfModelConfig;

use crate::error::{ExecutorError, ExecutorResult};
use crate::input_batch::{InputBatch, PreparedInputs};
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
    /// Batch sizes to capture as CUDA graphs (sorted, deduplicated).
    pub cuda_graph_sizes: Vec<usize>,
    /// Run cublasLt algorithm benchmarking during warmup (--cublas-autotune).
    pub cublas_autotune: bool,
    /// Fraction of GPU memory to use (0.0-1.0). Used to compute KV cache budget.
    pub gpu_memory_utilization: f64,
    /// Pooling strategy: "auto", "last", "cls", "mean".
    pub pooling_strategy: String,
    /// Whether the worker runs in pooling mode (--runner pooling).
    pub is_pooling: bool,
}

// ---------------------------------------------------------------------------
// Model enum (dispatches to LLaMA / Qwen2 / Gemma2)
// ---------------------------------------------------------------------------

/// Supported model architectures in the vllm-cuda backend.
enum CudaModel {
    Llama(vllm_cuda::model::llama::LlamaForCausalLM),
    Qwen2(vllm_cuda::model::qwen2::Qwen2ForCausalLM),
    Gemma2(vllm_cuda::model::gemma2::Gemma2ForCausalLM),
    Gemma3(vllm_cuda::model::gemma3::Gemma3ForCausalLM),
}

impl CudaModel {
    fn num_layers(&self) -> usize {
        match self {
            Self::Llama(m) => m.model.layers.len(),
            Self::Qwen2(m) => m.0.model.layers.len(),
            Self::Gemma2(m) => m.model.layers.len(),
            Self::Gemma3(m) => m.model.layers.len(),
        }
    }

    fn num_kv_heads(&self) -> usize {
        match self {
            Self::Llama(m) => m.model.layers[0].self_attn.num_kv_heads,
            Self::Qwen2(m) => m.0.model.layers[0].self_attn.num_kv_heads,
            Self::Gemma2(m) => m.model.layers[0].self_attn.num_kv_heads,
            Self::Gemma3(m) => m.model.layers[0].self_attn.num_kv_heads,
        }
    }

    fn head_dim(&self) -> usize {
        match self {
            Self::Llama(m) => m.model.layers[0].self_attn.head_dim,
            Self::Qwen2(m) => m.0.model.layers[0].self_attn.head_dim,
            Self::Gemma2(m) => m.model.layers[0].self_attn.head_dim,
            Self::Gemma3(m) => m.model.layers[0].self_attn.head_dim,
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Llama(m) => m.lm_head.out_features(),
            Self::Qwen2(m) => m.0.lm_head.out_features(),
            Self::Gemma2(m) => m.lm_head.out_features(),
            Self::Gemma3(m) => m.lm_head.out_features(),
        }
    }

    fn hidden_size(&self) -> usize {
        match self {
            Self::Llama(m) => m.lm_head.in_features(),
            Self::Qwen2(m) => m.0.lm_head.in_features(),
            Self::Gemma2(m) => m.lm_head.in_features(),
            Self::Gemma3(m) => m.lm_head.in_features(),
        }
    }

    /// Run backbone forward pass (without lm_head), returning hidden states
    /// `[num_tokens, hidden_size]` on GPU.
    ///
    /// # Safety
    /// All GpuTensors must be valid. CUDA context must be current.
    #[allow(clippy::too_many_arguments)]
    unsafe fn hidden_states(
        &self,
        input_ids: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
    ) -> GpuTensor {
        match self {
            Self::Llama(m) => unsafe {
                m.model.forward_owned(
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                )
            },
            Self::Qwen2(m) => unsafe {
                m.0.model.forward_owned(
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                )
            },
            Self::Gemma2(m) => unsafe {
                m.model.forward_owned(
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                )
            },
            Self::Gemma3(m) => unsafe {
                m.model.forward_owned(
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                )
            },
        }
    }

    /// Run forward pass, returning logits `[num_reqs, vocab_size]`.
    ///
    /// # Safety
    /// All GpuTensors must be valid. CUDA context must be current.
    #[allow(dead_code, clippy::too_many_arguments)]
    unsafe fn forward(
        &self,
        input_ids: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
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
                    seqused_k,
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
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                    last_token_indices,
                )
            },
            Self::Gemma2(m) => unsafe {
                m.forward_owned(
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                    last_token_indices,
                )
            },
            Self::Gemma3(m) => unsafe {
                m.forward_owned(
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
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

    /// Forward using caching allocator (zero D2D copies between layers).
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward_owned(
        &self,
        input_ids: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
        last_token_indices: Option<GpuTensor>,
    ) -> GpuTensor {
        match self {
            Self::Llama(m) => unsafe {
                m.forward_owned(
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                    last_token_indices,
                )
            },
            Self::Qwen2(m) => unsafe {
                m.forward_owned(
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                    last_token_indices,
                )
            },
            Self::Gemma2(m) => unsafe {
                m.forward_owned(
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                    last_token_indices,
                )
            },
            Self::Gemma3(m) => unsafe {
                m.forward_owned(
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
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

    // Parse llama3 rope_scaling if present.
    let llama3_rope_scaling = hf.extra.get("rope_scaling").and_then(|rs| {
        let rope_type = rs
            .get("rope_type")
            .or_else(|| rs.get("type"))
            .and_then(|v| v.as_str())?;
        if rope_type != "llama3" {
            return None;
        }
        Some(vllm_cuda::model::llama::Llama3RopeScaling {
            factor: rs.get("factor")?.as_f64()?,
            low_freq_factor: rs
                .get("low_freq_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0),
            high_freq_factor: rs
                .get("high_freq_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(4.0),
            original_max_position_embeddings: rs
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(8192) as usize,
        })
    });

    if let Some(ref s) = llama3_rope_scaling {
        info!(
            "llama3 rope_scaling: factor={}, low_freq={}, high_freq={}, orig_max={}",
            s.factor, s.low_freq_factor, s.high_freq_factor, s.original_max_position_embeddings
        );
    }

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
        llama3_rope_scaling,
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

fn gemma3_config_from_hf(
    hf: &HfModelConfig,
) -> ExecutorResult<vllm_cuda::model::gemma3::Gemma3Config> {
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
    let sliding_window = hf
        .extra
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let sliding_window_pattern = hf
        .extra
        .get("sliding_window_pattern")
        .and_then(|v| v.as_u64())
        .unwrap_or(6) as usize;
    let rope_local_base_freq = hf
        .extra
        .get("rope_local_base_freq")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0);
    let attention_bias = hf
        .extra
        .get("attention_bias")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Gemma3: every Nth layer is global, rest are sliding.
    // Python: layer_types[i] == "sliding_attention" when (i+1) % sliding_window_pattern != 0
    let layer_is_sliding: Vec<bool> = (0..num_hidden_layers)
        .map(|i| (i + 1) % sliding_window_pattern != 0)
        .collect();

    Ok(vllm_cuda::model::gemma3::Gemma3Config {
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        num_hidden_layers,
        intermediate_size: hf.intermediate_size.unwrap_or(hidden_size * 4),
        vocab_size: hf.vocab_size.unwrap_or(262144),
        max_position_embeddings: hf.max_position_embeddings.unwrap_or(131072),
        rms_norm_eps: hf.rms_norm_eps.unwrap_or(1e-6) as f32,
        rope_theta: hf.rope_theta.unwrap_or(1_000_000.0),
        rope_local_base_freq,
        head_dim,
        query_pre_attn_scalar,
        tie_word_embeddings: hf.tie_word_embeddings.unwrap_or(true),
        layer_is_sliding,
        sliding_window,
        attention_bias,
    })
}

// ---------------------------------------------------------------------------
// Pinned host staging buffers
// ---------------------------------------------------------------------------

/// Pre-allocated pinned (page-locked) host buffers for CUDA graph replay.
///
/// Eliminates per-step heap Vec allocations and enables true async DMA
/// (pageable memory forces the CUDA driver to stage through an internal
/// pinned buffer, serializing the transfer).
struct HostStaging {
    /// `[max_batch]` u32 — input token IDs.
    input_ids: PinnedBuf,
    /// `[max_batch]` u32 — position indices.
    positions: PinnedBuf,
    /// `[max_batch]` i64 — KV cache slot mapping.
    slot_mapping: PinnedBuf,
    /// `[max_batch + 1]` i32 — cumulative query sequence lengths.
    cu_seqlens_q: PinnedBuf,
    /// `[max_batch]` i32 — per-sequence K lengths for paged FA2.
    seqused_k: PinnedBuf,
    /// `[max_batch * GRAPH_MAX_BLOCKS_PER_SEQ]` i32 — page table.
    block_table: PinnedBuf,
    /// `[max_batch]` u32 — D2H token IDs from graph argmax.
    host_token_ids: PinnedBuf,
    /// `[max_batch * 5]` f32 — packed sampling params (temps, top_ks, top_ps, min_ps, randoms).
    sampling_packed: PinnedBuf,
}

impl HostStaging {
    /// Allocate pinned staging buffers for up to `max_batch` decode requests.
    ///
    /// # Safety
    /// Requires active CUDA context.
    unsafe fn new(max_batch: usize) -> anyhow::Result<Self> {
        Ok(Self {
            input_ids: unsafe { PinnedBuf::new(max_batch * 4)? },
            positions: unsafe { PinnedBuf::new(max_batch * 4)? },
            slot_mapping: unsafe { PinnedBuf::new(max_batch * 8)? },
            cu_seqlens_q: unsafe { PinnedBuf::new((max_batch + 1) * 4)? },
            seqused_k: unsafe { PinnedBuf::new(max_batch * 4)? },
            block_table: unsafe { PinnedBuf::new(max_batch * GRAPH_MAX_BLOCKS_PER_SEQ * 4)? },
            host_token_ids: unsafe { PinnedBuf::new(max_batch * 4)? },
            sampling_packed: unsafe { PinnedBuf::new(max_batch * 5 * 4)? },
        })
    }

    /// Fill block_table pinned buffer from attention metadata block_ids.
    /// Returns a slice of the pinned buffer with `graph_bs * GRAPH_MAX_BLOCKS_PER_SEQ` elements.
    unsafe fn fill_block_table<'a>(
        &'a self,
        block_ids: &[Vec<usize>],
        graph_bs: usize,
    ) -> &'a [i32] {
        let n = graph_bs * GRAPH_MAX_BLOCKS_PER_SEQ;
        let bt = unsafe { self.block_table.slice_mut::<i32>(n) };
        bt.fill(0);
        for (i, blocks) in block_ids.iter().enumerate() {
            for (j, &bid) in blocks.iter().enumerate() {
                if j < GRAPH_MAX_BLOCKS_PER_SEQ {
                    bt[i * GRAPH_MAX_BLOCKS_PER_SEQ + j] = bid as i32;
                }
            }
        }
        &bt[..n]
    }
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
    /// CUDA graph runner for decode batches.
    graph_runner: Option<CudaGraphRunner>,
    /// CUDA graph runner for single-sequence prefill batches.
    prefill_graph_runner: Option<PrefillGraphRunner>,
    /// Batch size of the last graph replay. When the batch composition is
    /// unchanged, we can skip the input_ids H2D copy because the graph's
    /// D2D scatter already placed argmax results into the persistent buffer.
    last_graph_batch_size: Option<usize>,
    /// True when persistent graph buffers contain valid metadata from a
    /// previous replay (positions, slot_mapping, seqused_k). When true,
    /// we use `replay_decode_fast()` which updates metadata on GPU instead
    /// of building Vecs on CPU and doing H2D copies.
    graph_metadata_valid: bool,

    /// Pre-allocated pinned host staging buffers for graph replay.
    /// Initialized after graph capture in `compile_or_warm_up_model`.
    host_staging: Option<HostStaging>,

    // Per-request state (mirrors CandleWorker).
    token_buffers: HashMap<String, Vec<u32>>,
    sampling_params_map: HashMap<String, SamplingParams>,
    input_batch: InputBatch,
    preloaded_tokenizer: Option<tokenizers::Tokenizer>,
    /// Set once per thread to avoid redundant `ctx_set_current` driver calls.
    ctx_set_on_thread: bool,
    /// Resolved pooling strategy for embedding mode.
    pooling_strategy: vllm_models::embedding::PoolingStrategy,
    /// Whether executing in pooling mode (--runner pooling).
    is_pooling: bool,

    /// Per-request grammar guide state for constrained decoding.
    #[cfg(feature = "guided-decoding")]
    grammar_states: HashMap<String, vllm_models::grammar::GrammarGuide>,
    /// Compiled vocabulary for grammar-guided decoding (built once from tokenizer).
    #[cfg(feature = "guided-decoding")]
    grammar_vocabulary: Option<outlines_core::vocabulary::Vocabulary>,
}

unsafe impl Send for CudaWorker {}

impl CudaWorker {
    pub fn new(config: CudaWorkerConfig) -> Self {
        let is_pooling = config.is_pooling;
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
            graph_runner: None,
            prefill_graph_runner: None,
            last_graph_batch_size: None,
            graph_metadata_valid: false,
            host_staging: None,
            token_buffers: HashMap::new(),
            sampling_params_map: HashMap::new(),
            input_batch: InputBatch::new(),
            preloaded_tokenizer: None,
            ctx_set_on_thread: false,
            pooling_strategy: vllm_models::embedding::PoolingStrategy::Last,
            is_pooling,
            #[cfg(feature = "guided-decoding")]
            grammar_states: HashMap::new(),
            #[cfg(feature = "guided-decoding")]
            grammar_vocabulary: None,
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

    /// Build grammar vocabulary on demand (lazy — deferred from startup).
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
            info!("CudaWorker: no tokenizer.json found, grammar-guided decoding unavailable");
            return;
        }
        let tokenizer = match tokenizers::Tokenizer::from_file(&tokenizer_path) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(
                    "CudaWorker: failed to load tokenizer.json for grammar vocabulary: {e}"
                );
                return;
            }
        };

        let hf_vocab = tokenizer.get_vocab(true);
        let eos_token_id = tokenizer.token_to_id("</s>").unwrap_or(0);
        let tokens: Vec<(u32, String)> = hf_vocab.into_iter().map(|(s, id)| (id, s)).collect();

        match vllm_models::grammar::build_vocabulary(&tokens, eos_token_id) {
            Ok(vocab) => {
                info!(
                    "CudaWorker: grammar vocabulary built ({} tokens, eos={})",
                    vocab.len(),
                    eos_token_id
                );
                self.grammar_vocabulary = Some(vocab);
            }
            Err(e) => {
                tracing::warn!("CudaWorker: failed to build grammar vocabulary: {e}");
            }
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

    /// H2D copy a u32 slice into a caching-allocator tensor.
    fn h2d_u32(data: &[u32], device: &mut GpuDevice) -> ExecutorResult<GpuTensor> {
        let t = device.caching.alloc_tensor(&[data.len()], GpuDType::U32);
        unsafe {
            driver::memcpy_htod_async(
                t.as_gpu_tensor().raw_ptr(),
                data.as_ptr() as *const u8,
                data.len() * 4,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("H2D u32: {e}")))?;
        Ok(t.into_gpu_tensor())
    }

    /// H2D copy an i32 slice into a caching-allocator tensor.
    fn h2d_i32(data: &[i32], device: &mut GpuDevice) -> ExecutorResult<GpuTensor> {
        let t = device.caching.alloc_tensor(&[data.len()], GpuDType::I32);
        unsafe {
            driver::memcpy_htod_async(
                t.as_gpu_tensor().raw_ptr(),
                data.as_ptr() as *const u8,
                data.len() * 4,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("H2D i32: {e}")))?;
        Ok(t.into_gpu_tensor())
    }

    fn h2d_i64(data: &[i64], device: &mut GpuDevice) -> ExecutorResult<GpuTensor> {
        let t = device.caching.alloc_tensor(&[data.len()], GpuDType::I64);
        unsafe {
            driver::memcpy_htod_async(
                t.as_gpu_tensor().raw_ptr(),
                data.as_ptr() as *const u8,
                data.len() * 8,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("H2D i64: {e}")))?;
        Ok(t.into_gpu_tensor())
    }

    fn h2d_f32(data: &[f32], device: &mut GpuDevice) -> ExecutorResult<GpuTensor> {
        let t = device.caching.alloc_tensor(&[data.len()], GpuDType::F32);
        unsafe {
            driver::memcpy_htod_async(
                t.as_gpu_tensor().raw_ptr(),
                data.as_ptr() as *const u8,
                data.len() * 4,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("H2D f32: {e}")))?;
        Ok(t.into_gpu_tensor())
    }

    /// Build attention metadata tensors from `AttentionMetadata`.
    ///
    /// Returns `(slot_mapping, cu_seqlens_q, seqused_k, block_table, max_seqlen_q, max_seqlen_k)`.
    /// `seqused_k` has per-sequence K lengths `[num_reqs]` for the paged FA2 splitkv kernel.
    fn build_attention_tensors(
        meta: &vllm_models::AttentionMetadata,
        block_size: usize,
        device: &mut GpuDevice,
    ) -> ExecutorResult<(GpuTensor, GpuTensor, GpuTensor, GpuTensor, usize, usize)> {
        let num_reqs = meta.num_reqs;

        // cu_seqlens_q: cumulative query lengths [num_reqs + 1].
        let cu_seqlens_q: Vec<i32> = meta.query_start_loc.iter().map(|&x| x as i32).collect();
        let gpu_cu_seqlens_q = Self::h2d_i32(&cu_seqlens_q, device)?;

        // seqused_k: per-sequence K lengths [num_reqs] (used by paged FA2 splitkv kernel).
        let seqused_k: Vec<i32> = meta.seq_lens.iter().map(|&sl| sl as i32).collect();
        let gpu_seqused_k = Self::h2d_i32(&seqused_k, device)?;

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

        // block_table: [num_reqs, max_blocks_per_seq] i32 (padded with 0).
        let max_blocks = meta.block_ids.iter().map(|b| b.len()).max().unwrap_or(0);
        let gpu_block_table = if max_blocks > 0 {
            let mut block_table = vec![0i32; num_reqs * max_blocks];
            for (i, blocks) in meta.block_ids.iter().enumerate() {
                for (j, &bid) in blocks.iter().enumerate() {
                    block_table[i * max_blocks + j] = bid as i32;
                }
            }
            let t = Self::h2d_i32(&block_table, device)?;
            unsafe { GpuTensor::new(t.raw_ptr(), &[num_reqs, max_blocks], GpuDType::I32) }
        } else {
            unsafe { GpuTensor::new(std::ptr::null_mut(), &[0, 0], GpuDType::I32) }
        };

        Ok((
            gpu_slot_mapping,
            gpu_cu_seqlens_q,
            gpu_seqused_k,
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

    /// Pool hidden states on GPU, D2H the small result, cast to f32, L2 normalize.
    ///
    /// `hidden_states` is `[num_tokens, hidden_size]` on GPU in model dtype.
    /// Returns `Vec<f32>` of length `hidden_size`, L2-normalized.
    fn pool_and_normalize(
        hidden_states: GpuTensor,
        num_tokens: usize,
        strategy: vllm_models::embedding::PoolingStrategy,
        device: &mut GpuDevice,
    ) -> ExecutorResult<Vec<f32>> {
        use vllm_models::embedding::PoolingStrategy;

        let hidden_size = hidden_states.dim(1);

        // 1. GPU-side pooling: extract [hidden_size] from [num_tokens, hidden_size].
        let pooled_gpu = match strategy {
            PoolingStrategy::Last => unsafe {
                vllm_cuda::kernels::pool_select_row(
                    hidden_states,
                    num_tokens - 1,
                    &mut device.caching,
                    device.compute_stream,
                )
            },
            PoolingStrategy::Cls => unsafe {
                vllm_cuda::kernels::pool_select_row(
                    hidden_states,
                    0,
                    &mut device.caching,
                    device.compute_stream,
                )
            },
            PoolingStrategy::Mean => {
                // Mean pooling needs f32 for cuBLAS gemv. For bf16/f16 inputs,
                // D2H the small pooled tensor and compute mean on CPU.
                // (Avoids needing a bf16→f32 cast kernel for this small vector.)
                // Actually, for Mean we need to average ALL rows, so D2H all rows
                // is expensive. Instead, D2H just the pooled row for Last/CLS,
                // but for Mean we need GPU computation.
                //
                // Strategy: if dtype is f32, use GPU gemv. Otherwise, D2H the
                // full hidden_states and compute mean on CPU. For typical embedding
                // use cases, seq_len * hidden_size * 2 bytes is manageable
                // (512 * 4096 * 2 = 4MB). This is the simple path.
                //
                // TODO: add bf16→f32 cast kernel for full GPU mean pooling.
                if hidden_states.dtype() == GpuDType::F32 {
                    unsafe {
                        vllm_cuda::kernels::pool_mean_f32(
                            hidden_states,
                            &device.cublas,
                            &mut device.caching,
                            device.compute_stream,
                        )
                    }
                } else {
                    // D2H all hidden states, compute mean on CPU.
                    let all_f32 = Self::logits_to_cpu(hidden_states, device)?;
                    let mut mean = vec![0.0f32; hidden_size];
                    let n = num_tokens as f32;
                    for row in 0..num_tokens {
                        let start = row * hidden_size;
                        for (j, val) in mean.iter_mut().enumerate() {
                            *val += all_f32[start + j] / n;
                        }
                    }
                    // L2 normalize on CPU.
                    let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for v in &mut mean {
                            *v /= norm;
                        }
                    }
                    return Ok(mean);
                }
            }
        };

        // 2. D2H the small [hidden_size] vector.
        let pooled_gpu_t = pooled_gpu.as_gpu_tensor();
        let nbytes = hidden_size * pooled_gpu_t.dtype().size_bytes();
        let mut host_buf = vec![0u8; nbytes];
        unsafe {
            driver::memcpy_dtoh_async(
                host_buf.as_mut_ptr(),
                pooled_gpu_t.raw_ptr() as *const u8,
                nbytes,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("D2H pooled: {e}")))?;
        unsafe { driver::stream_synchronize(device.compute_stream) }
            .map_err(|e| ExecutorError::WorkerExecution(format!("sync: {e}")))?;

        // 3. Cast to f32 on CPU.
        let f32_vec: Vec<f32> = match pooled_gpu_t.dtype() {
            GpuDType::F32 => {
                let ptr = host_buf.as_ptr() as *const f32;
                unsafe { std::slice::from_raw_parts(ptr, hidden_size) }.to_vec()
            }
            GpuDType::F16 => {
                let ptr = host_buf.as_ptr() as *const half::f16;
                let slice = unsafe { std::slice::from_raw_parts(ptr, hidden_size) };
                slice.iter().map(|v| v.to_f32()).collect()
            }
            GpuDType::BF16 => {
                let ptr = host_buf.as_ptr() as *const half::bf16;
                let slice = unsafe { std::slice::from_raw_parts(ptr, hidden_size) };
                slice.iter().map(|v| v.to_f32()).collect()
            }
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "unexpected dtype for pooled embedding".into(),
                ));
            }
        };

        // 4. L2 normalize on CPU (tiny vector, ~4096 floats = 16KB).
        let norm: f32 = f32_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            Ok(f32_vec.into_iter().map(|x| x / norm).collect())
        } else {
            Ok(f32_vec)
        }
    }

    /// D2H copy token IDs using pinned staging if available.
    fn d2h_token_ids_pinned(
        staging: Option<&HostStaging>,
        gpu_tensor: &GpuTensor,
        num_reqs: usize,
        device: &mut GpuDevice,
    ) -> ExecutorResult<Vec<u32>> {
        if let Some(stg) = staging {
            // Async D2H: compute→event→transfer_stream→D2H→event→sync event.
            // This syncs only on the D2H event, not the full compute stream,
            // matching Python's AsyncGPUModelRunnerOutput pattern.
            unsafe {
                device.async_d2h(
                    stg.host_token_ids.ptr(),
                    gpu_tensor.raw_ptr() as *const u8,
                    num_reqs * 4,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("async D2H token ids: {e}")))?;
            device
                .sync_d2h()
                .map_err(|e| ExecutorError::WorkerExecution(format!("sync d2h: {e}")))?;
            Ok(unsafe { stg.host_token_ids.slice::<u32>(num_reqs) }.to_vec())
        } else {
            let mut ids = vec![0u32; num_reqs];
            unsafe {
                device.async_d2h(
                    ids.as_mut_ptr() as *mut u8,
                    gpu_tensor.raw_ptr() as *const u8,
                    num_reqs * 4,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("async D2H token ids: {e}")))?;
            device
                .sync_d2h()
                .map_err(|e| ExecutorError::WorkerExecution(format!("sync d2h: {e}")))?;
            Ok(ids)
        }
    }
    /// Full GPU sampling pipeline: grammar mask → logit bias → penalties → sample → logprobs.
    /// No CPU fallback — everything stays on GPU, matching Python vLLM exactly.
    ///
    /// Takes individual field references to avoid double-borrow of `self` (since
    /// `device` is already borrowed from `self.device` by the caller).
    #[allow(clippy::too_many_arguments)]
    fn gpu_sample_and_finalize(
        sampling_params_map: &HashMap<String, SamplingParams>,
        #[cfg(feature = "guided-decoding")] grammar_states: &mut HashMap<
            String,
            vllm_models::grammar::GrammarGuide,
        >,
        host_staging: &Option<HostStaging>,
        input_batch: &mut InputBatch,
        token_buffers: &mut HashMap<String, Vec<u32>>,
        logits: GpuTensor,
        prepared: PreparedInputs,
        device: &mut GpuDevice,
        all_greedy: bool,
        vocab_size: usize,
    ) -> ExecutorResult<ModelRunnerOutput> {
        use rand::Rng;
        let num_reqs = prepared.req_inputs.len();

        // Determine which features are needed.
        let any_logprobs = prepared.req_inputs.iter().any(|r| {
            sampling_params_map
                .get(&r.req_id)
                .is_some_and(|p| p.logprobs.is_some())
        });
        let any_penalties = prepared.req_inputs.iter().any(|r| {
            sampling_params_map.get(&r.req_id).is_some_and(|p| {
                p.frequency_penalty != 0.0
                    || p.presence_penalty != 0.0
                    || p.repetition_penalty != 1.0
            })
        });
        let any_logit_bias = prepared.req_inputs.iter().any(|r| {
            sampling_params_map
                .get(&r.req_id)
                .is_some_and(|p| p.logit_bias.is_some())
        });
        #[cfg(feature = "guided-decoding")]
        let any_grammar = prepared
            .req_inputs
            .iter()
            .any(|r| grammar_states.contains_key(&r.req_id));
        #[cfg(not(feature = "guided-decoding"))]
        let any_grammar = false;

        let needs_f32 = any_penalties || any_logit_bias || any_grammar || any_logprobs;

        if !needs_f32 {
            // Fast path: no modifications needed, use native dtype sampling.
            if all_greedy {
                let token_ids_owned = unsafe {
                    vllm_cuda::kernels::argmax_batched(
                        logits,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };
                let token_ids_gpu = token_ids_owned.as_gpu_tensor();
                return Self::finalize_d2h_and_commit(
                    &token_ids_gpu,
                    num_reqs,
                    prepared,
                    device,
                    host_staging,
                    input_batch,
                    token_buffers,
                );
            }

            // Non-greedy fast path: Gumbel or full sampling on native dtype.
            let mut rng = rand::thread_rng();
            let all_no_filter = prepared.req_inputs.iter().all(|r| {
                sampling_params_map
                    .get(&r.req_id)
                    .is_none_or(|p| p.top_k <= 0 && p.top_p >= 1.0 && p.min_p <= 0.0)
            });

            let token_ids_owned = if all_no_filter {
                let stride = num_reqs * 4;
                let total_bytes = stride * 2;
                let packed_ptr = Self::get_sampling_packed_ptr_s(host_staging, total_bytes);
                let temps_ptr = packed_ptr as *mut f32;
                let randoms_ptr = unsafe { packed_ptr.add(stride) as *mut f32 };
                for (i, req_slice) in prepared.req_inputs.iter().enumerate() {
                    let params = sampling_params_map.get(&req_slice.req_id);
                    let t = params.map_or(1.0f32, |p| p.temperature.max(1e-7) as f32);
                    unsafe {
                        *temps_ptr.add(i) = t;
                        *randoms_ptr.add(i) = rng.r#gen::<f32>();
                    }
                }
                let gpu_packed_owned = device
                    .caching
                    .alloc_tensor(&[total_bytes / 4], GpuDType::F32);
                let gpu_packed = gpu_packed_owned.as_gpu_tensor();
                unsafe {
                    driver::memcpy_htod_async(
                        gpu_packed.raw_ptr(),
                        packed_ptr,
                        total_bytes,
                        device.compute_stream,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("H2D sampling: {e}")))?;
                let base = gpu_packed.raw_ptr();
                let gpu_temps = unsafe { GpuTensor::new(base, &[num_reqs], GpuDType::F32) };
                let gpu_randoms =
                    unsafe { GpuTensor::new(base.add(stride), &[num_reqs], GpuDType::F32) };
                unsafe {
                    vllm_cuda::kernels::sample_gumbel_batched(
                        logits,
                        gpu_temps,
                        gpu_randoms,
                        &mut device.caching,
                        device.compute_stream,
                    )
                }
            } else {
                let stride = num_reqs * 4;
                let total_bytes = stride * 5;
                let packed_ptr = Self::get_sampling_packed_ptr_s(host_staging, total_bytes);
                let temps_ptr = packed_ptr as *mut f32;
                let top_ks_ptr = unsafe { packed_ptr.add(stride) as *mut i32 };
                let top_ps_ptr = unsafe { packed_ptr.add(stride * 2) as *mut f32 };
                let min_ps_ptr = unsafe { packed_ptr.add(stride * 3) as *mut f32 };
                let randoms_ptr = unsafe { packed_ptr.add(stride * 4) as *mut f32 };
                for (i, req_slice) in prepared.req_inputs.iter().enumerate() {
                    let params = sampling_params_map.get(&req_slice.req_id);
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
                let gpu_packed_owned = device
                    .caching
                    .alloc_tensor(&[total_bytes / 4], GpuDType::F32);
                let gpu_packed = gpu_packed_owned.as_gpu_tensor();
                unsafe {
                    driver::memcpy_htod_async(
                        gpu_packed.raw_ptr(),
                        packed_ptr,
                        total_bytes,
                        device.compute_stream,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("H2D sampling: {e}")))?;
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
                unsafe {
                    vllm_cuda::kernels::sample_batched(
                        logits,
                        gpu_temps,
                        gpu_top_ks,
                        gpu_top_ps,
                        gpu_min_ps,
                        gpu_randoms,
                        &mut device.caching,
                        device.compute_stream,
                    )
                }
            };
            let token_ids_gpu = token_ids_owned.as_gpu_tensor();
            return Self::finalize_d2h_and_commit(
                &token_ids_gpu,
                num_reqs,
                prepared,
                device,
                host_staging,
                input_batch,
                token_buffers,
            );
        }

        // ---- Slow(er) path: cast to f32, apply modifications, sample on GPU ----

        // 1. Cast logits to f32.
        let logits_f32_owned = unsafe {
            vllm_cuda::kernels::cast_logits_to_f32(
                logits,
                &mut device.caching,
                device.compute_stream,
            )
        };
        let logits_f32 = logits_f32_owned.as_gpu_tensor();

        // 2. Save raw logits for logprobs (GPU copy, before modifications).
        let raw_logits_for_logprobs = if any_logprobs {
            let copy = device
                .caching
                .alloc_tensor(&[num_reqs, vocab_size], GpuDType::F32);
            unsafe {
                driver::memcpy_dtod_async(
                    copy.as_gpu_tensor().raw_ptr(),
                    logits_f32.raw_ptr() as *const u8,
                    num_reqs * vocab_size * 4,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("D2D logits copy: {e}")))?;
            Some(copy)
        } else {
            None
        };

        // 3. Apply grammar mask on GPU.
        #[cfg(feature = "guided-decoding")]
        if any_grammar {
            // Collect allowed token IDs per grammar request (CSR-packed).
            let mut allowed_ids_flat: Vec<i32> = Vec::new();
            let mut allowed_offsets: Vec<i32> = vec![0];
            let mut req_indices: Vec<i32> = Vec::new();

            for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
                if let Some(g) = grammar_states.get(&req_slice.req_id)
                    && let Some(allowed) = g.allowed_tokens()
                {
                    req_indices.push(req_idx as i32);
                    for &tok in &allowed {
                        allowed_ids_flat.push(tok as i32);
                    }
                    allowed_offsets.push(allowed_ids_flat.len() as i32);
                }
            }

            if !req_indices.is_empty() {
                // We need a backup of the logits before masking. If we already
                // have raw_logits_for_logprobs, reuse it; otherwise make one.
                // Keep `backup_owned` alive so the GPU memory isn't freed prematurely.
                let backup_owned = if raw_logits_for_logprobs.is_some() {
                    None
                } else {
                    let bk = device
                        .caching
                        .alloc_tensor(&[num_reqs, vocab_size], GpuDType::F32);
                    unsafe {
                        driver::memcpy_dtod_async(
                            bk.as_gpu_tensor().raw_ptr(),
                            logits_f32.raw_ptr() as *const u8,
                            num_reqs * vocab_size * 4,
                            device.compute_stream,
                        )
                    }
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("D2D grammar backup: {e}"))
                    })?;
                    Some(bk)
                };
                let backup = if let Some(ref raw) = raw_logits_for_logprobs {
                    raw.as_gpu_tensor()
                } else {
                    backup_owned.as_ref().unwrap().as_gpu_tensor()
                };

                let gpu_allowed = Self::h2d_i32(&allowed_ids_flat, device)?;
                let gpu_offsets = Self::h2d_i32(&allowed_offsets, device)?;
                let gpu_req_indices = Self::h2d_i32(&req_indices, device)?;

                unsafe {
                    vllm_cuda::kernels::apply_grammar_mask(
                        logits_f32,
                        backup,
                        gpu_allowed,
                        gpu_offsets,
                        gpu_req_indices,
                        device.compute_stream,
                    );
                }
            }
        }

        // 4. Apply logit bias on GPU.
        if any_logit_bias {
            let mut bias_ids_flat: Vec<i32> = Vec::new();
            let mut bias_vals_flat: Vec<f32> = Vec::new();
            let mut bias_offsets: Vec<i32> = vec![];

            for req_slice in &prepared.req_inputs {
                bias_offsets.push(bias_ids_flat.len() as i32);
                if let Some(params) = sampling_params_map.get(&req_slice.req_id)
                    && let Some(ref bias_map) = params.logit_bias
                {
                    for (&tok_id, &val) in bias_map {
                        bias_ids_flat.push(tok_id as i32);
                        bias_vals_flat.push(val);
                    }
                }
            }
            bias_offsets.push(bias_ids_flat.len() as i32);

            if !bias_ids_flat.is_empty() {
                let gpu_bias_ids = Self::h2d_i32(&bias_ids_flat, device)?;
                let gpu_bias_vals = Self::h2d_f32(&bias_vals_flat, device)?;
                let gpu_bias_offsets = Self::h2d_i32(&bias_offsets, device)?;

                unsafe {
                    vllm_cuda::kernels::apply_logit_bias(
                        logits_f32,
                        gpu_bias_ids,
                        gpu_bias_vals,
                        gpu_bias_offsets,
                        device.compute_stream,
                    );
                }
            }
        }

        // 5. Apply penalties on GPU.
        if any_penalties {
            // Build padded output_token_ids and prompt_token_ids arrays.
            let mut max_output_len = 0usize;
            for req_slice in &prepared.req_inputs {
                let buf = token_buffers.get(&req_slice.req_id);
                let params = sampling_params_map.get(&req_slice.req_id);
                if let Some(buf) = buf
                    && let Some(p) = params
                    && (p.frequency_penalty != 0.0
                        || p.presence_penalty != 0.0
                        || p.repetition_penalty != 1.0)
                {
                    max_output_len = max_output_len.max(buf.len());
                }
            }
            if max_output_len > 0 {
                let pad_val = vocab_size as i32; // sentinel: never matches any valid token
                let mut output_ids_flat = vec![pad_val; num_reqs * max_output_len];
                let mut rep_pens = vec![1.0f32; num_reqs];
                let mut freq_pens = vec![0.0f32; num_reqs];
                let mut pres_pens = vec![0.0f32; num_reqs];

                for (i, req_slice) in prepared.req_inputs.iter().enumerate() {
                    if let Some(params) = sampling_params_map.get(&req_slice.req_id) {
                        rep_pens[i] = params.repetition_penalty as f32;
                        freq_pens[i] = params.frequency_penalty as f32;
                        pres_pens[i] = params.presence_penalty as f32;
                    }
                    if let Some(buf) = token_buffers.get(&req_slice.req_id) {
                        let row = &mut output_ids_flat[i * max_output_len..];
                        for (j, &tok) in buf.iter().enumerate() {
                            if j < max_output_len {
                                row[j] = tok as i32;
                            }
                        }
                    }
                }

                // H2D all penalty data.
                let gpu_out_ids = Self::h2d_i32(&output_ids_flat, device)?;
                // Empty prompt_ids: 0 columns → just need a valid pointer.
                let dummy_prompt = vec![pad_val; num_reqs]; // 1 column, all padding
                let gpu_prompt_ids = Self::h2d_i32(&dummy_prompt, device)?;
                let gpu_rep = Self::h2d_f32(&rep_pens, device)?;
                let gpu_freq = Self::h2d_f32(&freq_pens, device)?;
                let gpu_pres = Self::h2d_f32(&pres_pens, device)?;

                // Reshape for kernel.
                let gpu_out_ids_2d = unsafe {
                    GpuTensor::new(
                        gpu_out_ids.raw_ptr(),
                        &[num_reqs, max_output_len],
                        GpuDType::U32,
                    )
                };
                let gpu_prompt_ids_2d = unsafe {
                    GpuTensor::new(gpu_prompt_ids.raw_ptr(), &[num_reqs, 1], GpuDType::U32)
                };

                unsafe {
                    vllm_cuda::kernels::apply_penalties(
                        logits_f32,
                        gpu_out_ids_2d,
                        gpu_prompt_ids_2d,
                        gpu_rep,
                        gpu_freq,
                        gpu_pres,
                        device.compute_stream,
                    );
                }
            }
        }

        // 6. Sample on GPU (from modified f32 logits).
        let mut rng = rand::thread_rng();
        let token_ids_owned = if all_greedy {
            // Argmax on f32 logits.
            unsafe {
                vllm_cuda::kernels::argmax_batched(
                    logits_f32,
                    &mut device.caching,
                    device.compute_stream,
                )
            }
        } else {
            // Full sampling on f32 logits.
            let stride = num_reqs * 4;
            let total_bytes = stride * 5;
            let packed_ptr = Self::get_sampling_packed_ptr_s(host_staging, total_bytes);
            let temps_ptr = packed_ptr as *mut f32;
            let top_ks_ptr = unsafe { packed_ptr.add(stride) as *mut i32 };
            let top_ps_ptr = unsafe { packed_ptr.add(stride * 2) as *mut f32 };
            let min_ps_ptr = unsafe { packed_ptr.add(stride * 3) as *mut f32 };
            let randoms_ptr = unsafe { packed_ptr.add(stride * 4) as *mut f32 };
            for (i, req_slice) in prepared.req_inputs.iter().enumerate() {
                let params = sampling_params_map.get(&req_slice.req_id);
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
            let gpu_packed_owned = device
                .caching
                .alloc_tensor(&[total_bytes / 4], GpuDType::F32);
            let gpu_packed = gpu_packed_owned.as_gpu_tensor();
            unsafe {
                driver::memcpy_htod_async(
                    gpu_packed.raw_ptr(),
                    packed_ptr,
                    total_bytes,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("H2D sampling: {e}")))?;
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
            unsafe {
                vllm_cuda::kernels::sample_batched(
                    logits_f32,
                    gpu_temps,
                    gpu_top_ks,
                    gpu_top_ps,
                    gpu_min_ps,
                    gpu_randoms,
                    &mut device.caching,
                    device.compute_stream,
                )
            }
        };
        let token_ids_gpu = token_ids_owned.as_gpu_tensor();

        // 7. Gather logprobs on GPU (if needed).
        let logprobs_output = if let Some(raw_logits_owned) = raw_logits_for_logprobs {
            let raw_logits = raw_logits_owned.as_gpu_tensor();
            // Find max num_logprobs across requests.
            let max_logprobs = prepared
                .req_inputs
                .iter()
                .filter_map(|r| sampling_params_map.get(&r.req_id).and_then(|p| p.logprobs))
                .max()
                .unwrap_or(0) as usize;

            if max_logprobs > 0 {
                let (topk_lp, topk_idx, topk_ranks) = unsafe {
                    vllm_cuda::kernels::log_softmax_topk_gather(
                        raw_logits,
                        token_ids_gpu,
                        max_logprobs,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };

                // D2H the small topk tensors.
                let k = max_logprobs + 1;
                let n_elements = num_reqs * k;
                let mut host_lp = vec![0.0f32; n_elements];
                let mut host_idx = vec![0i32; n_elements];
                let mut host_ranks = vec![0u32; n_elements];
                unsafe {
                    driver::memcpy_dtoh_async(
                        host_lp.as_mut_ptr() as *mut u8,
                        topk_lp.as_gpu_tensor().raw_ptr() as *const u8,
                        n_elements * 4,
                        device.compute_stream,
                    )
                    .map_err(|e| ExecutorError::WorkerExecution(format!("D2H logprobs: {e}")))?;
                    driver::memcpy_dtoh_async(
                        host_idx.as_mut_ptr() as *mut u8,
                        topk_idx.as_gpu_tensor().raw_ptr() as *const u8,
                        n_elements * 4,
                        device.compute_stream,
                    )
                    .map_err(|e| ExecutorError::WorkerExecution(format!("D2H logprob idx: {e}")))?;
                    driver::memcpy_dtoh_async(
                        host_ranks.as_mut_ptr() as *mut u8,
                        topk_ranks.as_gpu_tensor().raw_ptr() as *const u8,
                        n_elements * 4,
                        device.compute_stream,
                    )
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("D2H logprob ranks: {e}"))
                    })?;
                    driver::stream_synchronize(device.compute_stream)
                        .map_err(|e| ExecutorError::WorkerExecution(format!("sync: {e}")))?;
                }

                // Build per-request LogprobsOutput.
                let mut logprobs_map: HashMap<String, Vec<vllm_common::LogprobsOutput>> =
                    HashMap::new();
                for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
                    let requested_n = sampling_params_map
                        .get(&req_slice.req_id)
                        .and_then(|p| p.logprobs);
                    if let Some(n) = requested_n {
                        let n = n as usize;
                        let base = req_idx * k;
                        // Slot 0 = sampled token.
                        let sampled = vllm_common::TokenLogprob {
                            token_id: host_idx[base] as u32,
                            logprob: host_lp[base],
                            rank: host_ranks[base],
                        };
                        let mut top_logprobs = Vec::with_capacity(n);
                        for j in 1..=n.min(max_logprobs) {
                            top_logprobs.push(vllm_common::TokenLogprob {
                                token_id: host_idx[base + j] as u32,
                                logprob: host_lp[base + j],
                                rank: host_ranks[base + j],
                            });
                        }
                        logprobs_map
                            .entry(req_slice.req_id.clone())
                            .or_default()
                            .push(vllm_common::LogprobsOutput {
                                sampled,
                                top_logprobs,
                            });
                    }
                }
                Some(logprobs_map)
            } else {
                None
            }
        } else {
            None
        };

        // 8. D2H sampled tokens, commit, build output.
        let host_ids =
            Self::d2h_token_ids_pinned(host_staging.as_ref(), &token_ids_gpu, num_reqs, device)?;

        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
            let tok = host_ids[req_idx];

            // Grammar advance on CPU (Python also does FSM state on CPU).
            #[cfg(feature = "guided-decoding")]
            if let Some(g) = grammar_states.get_mut(&req_slice.req_id) {
                g.advance(tok);
            }

            input_batch.commit_step(
                &req_slice.req_id,
                &[tok],
                req_slice.token_count,
                !req_slice.spec_token_ids.is_empty(),
            );
            if let Some(buf) = token_buffers.get_mut(&req_slice.req_id) {
                buf.push(tok);
            }
        }

        let req_ids: Vec<String> = prepared
            .req_inputs
            .iter()
            .map(|r| r.req_id.clone())
            .collect();
        input_batch.reclaim_buffers(prepared);
        let mut output = ModelRunnerOutput::from_ordered(req_ids, host_ids);

        if let Some(mut logprobs_map) = logprobs_output {
            let logprobs_vec: Vec<Option<Vec<vllm_common::LogprobsOutput>>> = output
                .req_ids
                .iter()
                .map(|rid| logprobs_map.remove(rid))
                .collect();
            output.logprobs = Some(logprobs_vec);
        }

        Ok(output)
    }

    /// Get a pointer to pinned host memory for packing sampling parameters.
    fn get_sampling_packed_ptr_s(host_staging: &Option<HostStaging>, min_bytes: usize) -> *mut u8 {
        if let Some(stg) = host_staging {
            debug_assert!(min_bytes <= stg.sampling_packed.capacity_bytes());
            stg.sampling_packed.ptr()
        } else {
            vec![0u8; min_bytes].leak().as_mut_ptr()
        }
    }

    /// D2H + sync + commit_step helper (static to avoid borrow conflicts).
    fn finalize_d2h_and_commit(
        token_ids_gpu: &GpuTensor,
        num_reqs: usize,
        prepared: PreparedInputs,
        device: &mut GpuDevice,
        host_staging: &Option<HostStaging>,
        input_batch: &mut InputBatch,
        token_buffers: &mut HashMap<String, Vec<u32>>,
    ) -> ExecutorResult<ModelRunnerOutput> {
        let host_ids =
            Self::d2h_token_ids_pinned(host_staging.as_ref(), token_ids_gpu, num_reqs, device)?;
        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
            let tok = host_ids[req_idx];
            input_batch.commit_step(
                &req_slice.req_id,
                &[tok],
                req_slice.token_count,
                !req_slice.spec_token_ids.is_empty(),
            );
            if let Some(buf) = token_buffers.get_mut(&req_slice.req_id) {
                buf.push(tok);
            }
        }
        let req_ids: Vec<String> = prepared
            .req_inputs
            .iter()
            .map(|r| r.req_id.clone())
            .collect();
        input_batch.reclaim_buffers(prepared);
        Ok(ModelRunnerOutput::from_ordered(req_ids, host_ids))
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

        // 5. Parse weight files (CPU mmap — no GPU allocation yet).
        let mut weights = GpuWeights::from_dir(&model_dir, device.compute_stream)
            .map_err(|e| ExecutorError::WorkerInit(format!("weight load failed: {e}")))?;
        info!("CudaWorker: parsed {} weight tensors (CPU)", weights.len());

        // 6. Construct model based on architecture.
        let model = match arch.as_str() {
            "LlamaForCausalLM" | "MistralForCausalLM" | "Qwen3ForCausalLM" | "Phi3ForCausalLM" => {
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
            "Gemma3ForCausalLM" => {
                let config = gemma3_config_from_hf(&hf_config)?;
                let m = vllm_cuda::model::gemma3::Gemma3ForCausalLM::load(
                    &mut weights,
                    &config,
                    dtype,
                    device,
                )
                .map_err(|e| ExecutorError::WorkerInit(format!("Gemma3 load: {e}")))?;
                CudaModel::Gemma3(m)
            }
            "GraniteForCausalLM" => {
                let config = llama_config_from_hf(&hf_config)?;
                let mut m = vllm_cuda::model::llama::LlamaForCausalLM::load(
                    &mut weights,
                    &config,
                    dtype,
                    device,
                )
                .map_err(|e| ExecutorError::WorkerInit(format!("GraniteForCausalLM load: {e}")))?;

                // Parse Granite-specific multipliers from config.json extras.
                let extra = &hf_config.extra;
                let embedding_multiplier = extra
                    .get("embedding_multiplier")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32;
                let residual_multiplier = extra
                    .get("residual_multiplier")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32;
                let logits_scaling = extra
                    .get("logits_scaling")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32;
                let attention_multiplier = extra
                    .get("attention_multiplier")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(1.0 / (config.head_dim as f32).sqrt());

                // Apply multipliers.
                m.model.embedding_multiplier = embedding_multiplier;
                m.logits_scaling = logits_scaling;
                for layer in &mut m.model.layers {
                    layer.residual_multiplier = residual_multiplier;
                    layer.self_attn.scale = attention_multiplier;
                }

                info!(
                    "Granite multipliers: embedding={embedding_multiplier}, \
                     residual={residual_multiplier}, attention={attention_multiplier}, \
                     logits_scaling={logits_scaling}"
                );
                CudaModel::Llama(m)
            }
            _ => {
                return Err(ExecutorError::WorkerInit(format!(
                    "unsupported architecture for cuda-backend: {arch}. \
                     Supported: LlamaForCausalLM, MistralForCausalLM, Qwen3ForCausalLM, \
                     Phi3ForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM, Gemma3ForCausalLM, \
                     GraniteForCausalLM"
                )));
            }
        };

        // Sync to ensure all async H2D weight copies are complete.
        unsafe { driver::stream_synchronize(device.compute_stream) }
            .map_err(|e| ExecutorError::WorkerInit(format!("weight sync: {e}")))?;
        drop(weights); // CPU mmaps freed, GPU memory owned by model layers

        self.model_dtype = dtype;
        self.resolved_architecture = Some(arch);
        self.model = Some(model);
        self.model_dir = Some(model_dir.clone());
        self.hf_config = Some(hf_config);

        // Resolve pooling strategy.
        self.pooling_strategy = match self.config.pooling_strategy.as_str() {
            "last" => vllm_models::embedding::PoolingStrategy::Last,
            "cls" => vllm_models::embedding::PoolingStrategy::Cls,
            "mean" => vllm_models::embedding::PoolingStrategy::Mean,
            _ => {
                // "auto": detect from 1_Pooling/config.json, default to Last.
                vllm_models::embedding::detect_pooling_strategy(&model_dir)
                    .unwrap_or(vllm_models::embedding::PoolingStrategy::Last)
            }
        };

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

        // Like Python vLLM: profile peak activation memory with a dummy forward
        // pass, then subtract it from available memory for KV cache sizing.
        let (free_before, _total) = cudarc::driver::result::mem_get_info()
            .map_err(|e| ExecutorError::WorkerInit(format!("cuMemGetInfo: {e}")))?;
        info!(
            "CudaWorker: {:.0} MB free VRAM",
            free_before as f64 / 1_048_576.0
        );

        let device = self
            .device
            .as_mut()
            .ok_or_else(|| ExecutorError::WorkerInit("device not initialized".into()))?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("model not loaded".into()))?;

        // Run dummy forward with max_num_batched_tokens to measure peak activations.
        let prefill_tokens = self.config.max_num_batched_tokens;
        info!("Profiling activation memory with dummy forward ({prefill_tokens} tokens)...");

        // Allocate dummy inputs from caching allocator.
        let dummy_ids = device
            .caching
            .alloc_gpu_tensor(&[prefill_tokens], GpuDType::U32);
        let dummy_pos = device
            .caching
            .alloc_gpu_tensor(&[prefill_tokens], GpuDType::U32);
        let dummy_slots = device
            .caching
            .alloc_gpu_tensor(&[prefill_tokens], GpuDType::I64);
        unsafe {
            driver::memset_d8(
                dummy_slots.raw_ptr(),
                0,
                prefill_tokens * 8,
                device.compute_stream,
            )
            .ok();
        }
        let cu_q: Vec<u32> = vec![0, prefill_tokens as u32];
        let gpu_cu_q = device.caching.alloc_gpu_tensor(&[2], GpuDType::U32);
        unsafe {
            driver::memcpy_htod_async(
                gpu_cu_q.raw_ptr(),
                cu_q.as_ptr() as *const u8,
                8,
                device.compute_stream,
            )
            .ok();
        }
        let dummy_seqused = device.caching.alloc_gpu_tensor(&[1], GpuDType::U32);
        let dummy_bt = device.caching.alloc_gpu_tensor(&[1, 1], GpuDType::U32);
        unsafe {
            driver::memset_d8(dummy_bt.raw_ptr(), 0, 4, device.compute_stream).ok();
        }

        // We don't have KV cache yet, so create a tiny one for profiling.
        let dummy_kv = unsafe {
            vllm_cuda::KvCachePool::new(
                model.num_layers(),
                1, // 1 block
                self.config.block_size,
                model.num_kv_heads(),
                model.head_dim(),
                self.model_dtype,
            )
        }
        .map_err(|e| ExecutorError::WorkerInit(format!("dummy KvCachePool: {e}")))?;

        // Run the forward pass to warm up cuBLAS and measure peak memory.
        unsafe {
            let _ = model.forward_owned(
                dummy_ids,
                dummy_pos,
                dummy_slots,
                gpu_cu_q,
                dummy_seqused,
                dummy_bt,
                prefill_tokens,
                prefill_tokens,
                &dummy_kv,
                device,
                None,
            );
            let _ = driver::stream_synchronize(device.compute_stream);
        }

        // Measure memory after profile run — the difference is peak activations.
        let (free_after, _) = cudarc::driver::result::mem_get_info()
            .map_err(|e| ExecutorError::WorkerInit(format!("cuMemGetInfo: {e}")))?;

        // Free the dummy KV cache and all cached allocator blocks.
        drop(dummy_kv);
        unsafe { device.caching.free_leaked_blocks() };
        device.caching.trim();

        let peak_activation_bytes = free_before.saturating_sub(free_after);

        // Match Python vLLM's memory calculation exactly.
        //
        // Python's flow (gpu_worker.py + mem_utils.py):
        //   init_snapshot = MemorySnapshot taken BEFORE model load
        //   requested = total_memory * gpu_memory_utilization
        //   non_kv_cache = weights + peak_activations + non_torch + 150 MiB
        //   available_kv_bytes = requested - non_kv_cache
        //   num_blocks = available_kv_bytes / bytes_per_block  (no extra util)
        //
        // Our free_before is measured AFTER model load (before profile run).
        // So (total - free_before) captures model weights + CUDA context +
        // cuBLAS workspace — equivalent to Python's (weights + non_torch).
        let (_, total_memory) = cudarc::driver::result::mem_get_info()
            .map_err(|e| ExecutorError::WorkerInit(format!("cuMemGetInfo: {e}")))?;
        let weights_and_overhead = total_memory.saturating_sub(free_before);

        let utilization = self.config.gpu_memory_utilization;
        let available_kv_bytes = compute_available_kv_bytes(
            total_memory,
            weights_and_overhead,
            peak_activation_bytes,
            utilization,
        );

        info!(
            "Memory profiling: total={:.1} GiB, weights+overhead={:.1} GiB, \
             peak_activations={:.1} GiB",
            total_memory as f64 / 1_073_741_824.0,
            weights_and_overhead as f64 / 1_073_741_824.0,
            peak_activation_bytes as f64 / 1_073_741_824.0,
        );
        info!(
            "Available KV cache memory: {:.1} GiB \
             (requested={:.1} GiB [total*{:.2}] - non_kv={:.1} GiB)",
            available_kv_bytes as f64 / 1_073_741_824.0,
            (total_memory as f64 * utilization) / 1_073_741_824.0,
            utilization,
            (weights_and_overhead + peak_activation_bytes + 150 * 1024 * 1024) as f64
                / 1_073_741_824.0,
        );

        // Return the direct KV cache bytes. compute_num_blocks must NOT
        // apply gpu_memory_utilization again — it's already baked in.
        Ok(available_kv_bytes)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        self.execute_model_inner(scheduler_output)
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
        let capture_sizes = if self.config.cuda_graph_sizes.is_empty() {
            // Match Python vLLM's capture sizes: [1, 2, 4] + range(8, 256, 8) + range(256, 512+1, 16)
            let mut sizes = vec![1, 2, 4];
            let mut s = 8;
            while s < 256 {
                sizes.push(s);
                s += 8;
            }
            while s <= 512 {
                sizes.push(s);
                s += 16;
            }
            sizes
        } else {
            self.config.cuda_graph_sizes.clone()
        };
        let max_bs = *capture_sizes.iter().max().unwrap();

        // No arena pre-sizing needed — caching allocator manages memory
        // dynamically like PyTorch's CUDACachingAllocator.

        let mut runner = unsafe { CudaGraphRunner::new(max_bs, vocab_size, self.model_dtype) }
            .map_err(|e| ExecutorError::WorkerInit(format!("CudaGraphRunner::new: {e}")))?;

        // Begin private pool for ALL graph captures (like PyTorch's shared graph pool).
        // One pool is shared across all batch sizes so blocks are reused.
        device.caching.begin_allocate_to_pool();

        for &bs in &capture_sizes {
            info!("Capturing CUDA graph for batch_size={bs}...");
            let kv_ref = kv_cache;
            let model_ref = model;

            // Capture with max_seqlen_k padded to 2048. Paged FA2 uses per-sequence
            // lengths from seqused_k and iterates via block_table, so a large
            // max_seqlen_k just over-allocates workspace — correctness is maintained.
            // This avoids re-capture as sequences grow during inference.
            let padded_max_seqlen_k: usize = 2048;

            let result = unsafe {
                runner.capture(bs, device, |inputs, dev| {
                    model_ref.forward_owned(
                        inputs.input_ids,
                        inputs.positions,
                        inputs.slot_mapping,
                        inputs.cu_seqlens_q,
                        inputs.seqused_k,
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

        // End private pool after all captures. Blocks in the pool that are
        // still free are effectively owned by the captured graphs.
        device.caching.end_allocate_to_pool();

        if !runner.captured_sizes().is_empty() {
            info!(
                "CUDA graphs captured for batch sizes: {:?}",
                runner.captured_sizes()
            );
            // Allocate pinned host staging buffers sized for the largest captured graph.
            let staging_max_bs = *runner.captured_sizes().last().unwrap();
            match unsafe { HostStaging::new(staging_max_bs) } {
                Ok(staging) => {
                    info!(
                        "Pinned host staging allocated for max_batch={}",
                        staging_max_bs
                    );
                    self.host_staging = Some(staging);
                }
                Err(e) => {
                    tracing::warn!("Failed to allocate pinned staging: {e}");
                }
            }
            self.graph_runner = Some(runner);
        }

        // Capture prefill graphs for common single-sequence token counts.
        // These cover the common case of one prompt arriving at a time.
        let max_prefill_tokens = self.config.max_num_batched_tokens;
        let prefill_sizes: Vec<usize> = [128, 256, 512, 1024, 2048, 4096, 8192]
            .iter()
            .copied()
            .filter(|&s| s <= max_prefill_tokens)
            .collect();

        if !prefill_sizes.is_empty() {
            let max_prefill = *prefill_sizes.last().unwrap();
            match unsafe { PrefillGraphRunner::new(max_prefill, vocab_size, self.model_dtype) } {
                Ok(mut prefill_runner) => {
                    for &num_tokens in &prefill_sizes {
                        info!("Capturing prefill CUDA graph for num_tokens={num_tokens}...");
                        let kv_ref = kv_cache;
                        let model_ref = model;

                        let result = unsafe {
                            prefill_runner.capture(num_tokens, device, |inputs, dev| {
                                model_ref.forward_owned(
                                    inputs.input_ids,
                                    inputs.positions,
                                    inputs.slot_mapping,
                                    inputs.cu_seqlens_q,
                                    inputs.seqused_k,
                                    inputs.block_table,
                                    num_tokens, // max_seqlen_q
                                    num_tokens, // max_seqlen_k
                                    kv_ref,
                                    dev,
                                    Some(inputs.last_token_indices),
                                )
                            })
                        };

                        match result {
                            Ok(()) => {
                                info!("Prefill CUDA graph captured for num_tokens={num_tokens}")
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to capture prefill graph for {num_tokens}: {e}"
                                );
                            }
                        }
                    }

                    if !prefill_runner.captured_sizes().is_empty() {
                        info!(
                            "Prefill CUDA graphs captured for token counts: {:?}",
                            prefill_runner.captured_sizes()
                        );
                        self.prefill_graph_runner = Some(prefill_runner);
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to create PrefillGraphRunner: {e}");
                }
            }
        }

        if self.config.cublas_autotune {
            unsafe { device.cublas.benchmark_plans() };
        }

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

    fn embed(&mut self, token_id_seqs: &[&[u32]]) -> ExecutorResult<Vec<Vec<f32>>> {
        // Ensure CUDA context is current.
        if let Some(ref dev) = self.device {
            unsafe { driver::ctx_set_current(dev.ctx) }
                .map_err(|e| ExecutorError::WorkerExecution(format!("ctx_set_current: {e}")))?;
        }

        let (model, kv_cache, device) = match (&self.model, &self.kv_cache, &mut self.device) {
            (Some(m), Some(kv), Some(d)) => (m, kv, d),
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "model, KV cache, or device not initialized".into(),
                ));
            }
        };

        let block_size = self.config.block_size;
        let strategy = self.pooling_strategy;
        let mut results = Vec::with_capacity(token_id_seqs.len());

        for token_ids in token_id_seqs {
            let num_tokens = token_ids.len();
            if num_tokens == 0 {
                let hidden_size = model.hidden_size();
                results.push(vec![0.0f32; hidden_size]);
                continue;
            }

            // Build positions [0, 1, 2, ...].
            let positions: Vec<u32> = (0..num_tokens as u32).collect();

            // Upload inputs.
            let gpu_input_ids = Self::h2d_u32(token_ids, device)?;
            let gpu_positions = Self::h2d_u32(&positions, device)?;

            // Slot mapping: use block 0 sequentially.
            let slot_mapping: Vec<i64> = (0..num_tokens)
                .map(|t| {
                    let block_idx = t / block_size;
                    let offset = t % block_size;
                    (block_idx * block_size + offset) as i64
                })
                .collect();
            let gpu_slot_mapping = Self::h2d_i64(&slot_mapping, device)?;

            // Attention metadata: single sequence.
            let cu_seqlens_q = vec![0i32, num_tokens as i32];
            let gpu_cu_q = Self::h2d_i32(&cu_seqlens_q, device)?;
            let seqused_k = vec![num_tokens as i32];
            let gpu_seqused_k = Self::h2d_i32(&seqused_k, device)?;

            // Block table: [1, max_blocks].
            let max_blocks = num_tokens.div_ceil(block_size);
            let block_table: Vec<i32> = (0..max_blocks as i32).collect();
            let gpu_bt = Self::h2d_i32(&block_table, device)?;
            let gpu_bt =
                unsafe { GpuTensor::new(gpu_bt.raw_ptr(), &[1, max_blocks], GpuDType::I32) };

            // Forward pass (backbone only).
            let hidden_states = unsafe {
                model.hidden_states(
                    gpu_input_ids,
                    gpu_positions,
                    gpu_slot_mapping,
                    gpu_cu_q,
                    gpu_seqused_k,
                    gpu_bt,
                    num_tokens,
                    num_tokens,
                    kv_cache,
                    device,
                )
            };

            // Pool + normalize.
            let embedding = Self::pool_and_normalize(hidden_states, num_tokens, strategy, device)?;
            results.push(embedding);

            // Free leaked tensors from this iteration.
            unsafe { device.caching.free_leaked_blocks() };
        }

        Ok(results)
    }
} // end impl Worker for CudaWorker

impl CudaWorker {
    fn execute_model_inner(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        // Ensure CUDA context is current on this thread (once per thread).
        if !self.ctx_set_on_thread {
            if let Some(ref dev) = self.device {
                unsafe { driver::ctx_set_current(dev.ctx) }
                    .map_err(|e| ExecutorError::WorkerExecution(format!("ctx_set_current: {e}")))?;
            }
            self.ctx_set_on_thread = true;
        }

        let block_size = self.config.block_size;

        // Free any leaked GPU tensors from the previous step (h2d inputs,
        // logits, sampling outputs, etc.). Keep graph output addresses pinned.
        if let Some(ref mut dev) = self.device {
            let mut keep: Vec<*const u8> = Vec::new();
            if let Some(ref runner) = self.graph_runner {
                keep.extend(runner.pinned_addresses());
            }
            unsafe { dev.caching.free_leaked_blocks_except(&keep) };
        }

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
            #[cfg(feature = "guided-decoding")]
            self.grammar_states.remove(req_id);
        }
        self.input_batch
            .remove_finished(&scheduler_output.finished_req_ids);

        // Ensure grammar vocabulary is built if any new request needs it.
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
                #[cfg(feature = "guided-decoding")]
                if let Some(ref grammar) = params.guided_grammar
                    && let Some(ref vocab) = self.grammar_vocabulary
                {
                    match vllm_models::grammar::GrammarGuide::from_guided_grammar(grammar, vocab) {
                        Ok(guide) => {
                            self.grammar_states.insert(new_req.req_id.clone(), guide);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to compile grammar for request {}: {e}",
                                new_req.req_id
                            );
                        }
                    }
                }
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
        let num_reqs = prepared.req_inputs.len();
        let total_tokens = prepared.flat_token_ids.len();

        // --- Pooling mode: run backbone, pool, return embeddings ---
        if self.is_pooling {
            let gpu_input_ids = Self::h2d_u32(&prepared.flat_token_ids, device)?;
            let gpu_positions = Self::h2d_u32(&prepared.flat_positions, device)?;

            let (
                slot_mapping,
                cu_seqlens_q,
                seqused_k,
                block_table_gpu,
                max_seqlen_q,
                max_seqlen_k,
            ) = Self::build_attention_tensors(&prepared.attn_meta, block_size, device)?;

            let hidden_states = unsafe {
                model.hidden_states(
                    gpu_input_ids,
                    gpu_positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table_gpu,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                )
            };

            // Pool each request's hidden states slice.
            let strategy = self.pooling_strategy;
            let meta = &prepared.attn_meta;
            let mut pooler_map: HashMap<String, Vec<f32>> = HashMap::new();

            for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
                let q_start = meta.query_start_loc[req_idx];
                let q_len = meta.q_lens[req_idx];
                let hidden_size = model.hidden_size();

                // Narrow hidden_states to this request's rows.
                let row_bytes = hidden_size * hidden_states.dtype().size_bytes();
                let req_hs = unsafe {
                    GpuTensor::new(
                        hidden_states.raw_ptr().add(q_start * row_bytes),
                        &[q_len, hidden_size],
                        hidden_states.dtype(),
                    )
                };

                let embedding = Self::pool_and_normalize(req_hs, q_len, strategy, device)?;
                pooler_map.insert(req_slice.req_id.clone(), embedding);

                // Commit step so InputBatch tracks progress.
                self.input_batch.commit_step(
                    &req_slice.req_id,
                    &[0], // dummy token — pooling doesn't generate tokens
                    req_slice.token_count,
                    false,
                );
            }

            self.input_batch.reclaim_buffers(prepared);

            // Build a ModelRunnerOutput with pooler_output and empty generation fields.
            let mut output = ModelRunnerOutput::from_token_map(HashMap::new());
            output.pooler_output = Some(pooler_map);
            return Ok(output);
        }

        let vocab_size = model.vocab_size();

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

        if is_decode && !use_graph {
            tracing::debug!("CUDA graph miss: decode bs={num_reqs} has no matching graph");
        }

        // Check if all requests are greedy (temp < 1e-6). Used to decide
        // whether to use the in-graph argmax fast path.
        let all_greedy = prepared.req_inputs.iter().all(|r| {
            self.sampling_params_map
                .get(&r.req_id)
                .is_none_or(|p| p.temperature < 1e-6)
        });

        // ---------------------------------------------------------------------------
        // Mixed batch: split into decode (CUDA graph) + prefill (eager) passes.
        // This avoids running the entire batch through the slow eager path when
        // most requests are decode (q_len=1) but a few are prefill chunks.
        // ---------------------------------------------------------------------------
        let has_decode = prepared.attn_meta.q_lens.contains(&1);
        let is_mixed = !is_decode && has_decode;
        let decode_graph_bs = if is_mixed {
            self.graph_runner.as_ref().and_then(|r| {
                let n_decode = prepared
                    .attn_meta
                    .q_lens
                    .iter()
                    .filter(|&&q| q == 1)
                    .count();
                r.nearest_graph_size(n_decode)
            })
        } else {
            None
        };

        if let Some(decode_graph_bs) = decode_graph_bs {
            // Partition requests into decode (q_len=1) and prefill (q_len>1).
            let mut decode_indices: Vec<usize> = Vec::new();
            let mut prefill_indices: Vec<usize> = Vec::new();
            for (i, &q) in prepared.attn_meta.q_lens.iter().enumerate() {
                if q == 1 {
                    decode_indices.push(i);
                } else {
                    prefill_indices.push(i);
                }
            }
            let n_decode = decode_indices.len();
            let n_prefill = prefill_indices.len();

            // --- Decode pass: run through CUDA graph ---
            // Build decode inputs from the subset of requests.
            let decode_logits = {
                let meta = &prepared.attn_meta;

                let mut input_ids = Vec::with_capacity(decode_graph_bs);
                let mut positions = Vec::with_capacity(decode_graph_bs);
                let mut slot_mapping: Vec<i64> = Vec::with_capacity(decode_graph_bs);
                let mut cu_seqlens_q: Vec<i32> = Vec::with_capacity(decode_graph_bs + 1);
                let mut seqused_k: Vec<i32> = Vec::with_capacity(decode_graph_bs);
                let mut block_table = vec![0i32; decode_graph_bs * GRAPH_MAX_BLOCKS_PER_SEQ];

                cu_seqlens_q.push(0);
                for (out_idx, &orig_idx) in decode_indices.iter().enumerate() {
                    // Each decode request has exactly 1 token.
                    let token_start = meta.query_start_loc[orig_idx];
                    input_ids.push(prepared.flat_token_ids[token_start]);
                    positions.push(prepared.flat_positions[token_start]);
                    cu_seqlens_q.push((out_idx + 1) as i32);
                    seqused_k.push(meta.seq_lens[orig_idx] as i32);

                    // Slot mapping: compute physical slot for this decode token.
                    let abs_pos = meta.tokens_before[orig_idx];
                    let block_idx = abs_pos / block_size;
                    let offset = abs_pos % block_size;
                    let block_ids = &meta.block_ids[orig_idx];
                    if block_idx < block_ids.len() {
                        slot_mapping.push((block_ids[block_idx] * block_size + offset) as i64);
                    } else {
                        slot_mapping.push(-1i64);
                    }

                    // Block table row.
                    for (j, &bid) in block_ids.iter().enumerate() {
                        if j < GRAPH_MAX_BLOCKS_PER_SEQ {
                            block_table[out_idx * GRAPH_MAX_BLOCKS_PER_SEQ + j] = bid as i32;
                        }
                    }
                }

                // Pad to graph batch size.
                input_ids.resize(decode_graph_bs, 0);
                positions.resize(decode_graph_bs, 0);
                slot_mapping.resize(decode_graph_bs, -1i64);
                for _ in n_decode..decode_graph_bs {
                    cu_seqlens_q.push(n_decode as i32);
                }
                seqused_k.resize(decode_graph_bs, 1);

                let runner = self.graph_runner.as_ref().unwrap();
                let replay_out = unsafe {
                    runner.replay(
                        decode_graph_bs,
                        &input_ids,
                        &positions,
                        &slot_mapping,
                        &cu_seqlens_q,
                        &seqused_k,
                        &block_table,
                        device,
                        false,
                    )
                }
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("mixed decode graph replay: {e}"))
                })?;

                // Slice to real decode requests (discard padding rows).
                if decode_graph_bs > n_decode {
                    replay_out.logits.narrow_dim0(0, n_decode)
                } else {
                    replay_out.logits
                }
            };

            // --- Prefill pass: run through eager forward ---
            let prefill_logits = {
                let meta = &prepared.attn_meta;

                // Build flat token/position arrays for prefill requests.
                let mut pf_token_ids: Vec<u32> = Vec::new();
                let mut pf_positions: Vec<u32> = Vec::new();
                let mut pf_q_lens: Vec<usize> = Vec::new();
                let mut pf_seq_lens: Vec<usize> = Vec::new();
                let mut pf_block_ids: Vec<Vec<usize>> = Vec::new();
                let mut pf_tokens_before: Vec<usize> = Vec::new();
                let mut pf_query_start_loc: Vec<usize> = vec![0];

                let mut pf_offset = 0usize;
                for &orig_idx in &prefill_indices {
                    let q_len = meta.q_lens[orig_idx];
                    let token_start = meta.query_start_loc[orig_idx];
                    pf_token_ids.extend_from_slice(
                        &prepared.flat_token_ids[token_start..token_start + q_len],
                    );
                    pf_positions.extend_from_slice(
                        &prepared.flat_positions[token_start..token_start + q_len],
                    );
                    pf_q_lens.push(q_len);
                    pf_seq_lens.push(meta.seq_lens[orig_idx]);
                    pf_block_ids.push(meta.block_ids[orig_idx].clone());
                    pf_tokens_before.push(meta.tokens_before[orig_idx]);
                    pf_offset += q_len;
                    pf_query_start_loc.push(pf_offset);
                }
                let pf_total_tokens = pf_token_ids.len();

                let pf_meta = vllm_models::AttentionMetadata::new(
                    n_prefill,
                    pf_total_tokens,
                    pf_query_start_loc,
                    pf_q_lens,
                    pf_seq_lens,
                    pf_block_ids,
                    pf_tokens_before,
                    vec![true; n_prefill],
                    prefill_indices
                        .iter()
                        .map(|&i| meta.req_ids[i].clone())
                        .collect(),
                );

                let gpu_input_ids = Self::h2d_u32(&pf_token_ids, device)?;
                let gpu_positions = Self::h2d_u32(&pf_positions, device)?;

                let (
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table_gpu,
                    max_seqlen_q,
                    max_seqlen_k,
                ) = Self::build_attention_tensors(&pf_meta, block_size, device)?;

                // last_token_indices: for each prefill request, index of last token in flat array.
                let last_token_indices = if n_prefill < pf_total_tokens {
                    let mut indices = Vec::with_capacity(n_prefill);
                    let mut off = 0u32;
                    for &q in &pf_meta.q_lens {
                        indices.push(off + q as u32 - 1);
                        off += q as u32;
                    }
                    Some(Self::h2d_u32(&indices, device)?)
                } else {
                    None
                };

                unsafe {
                    model.forward_owned(
                        gpu_input_ids,
                        gpu_positions,
                        slot_mapping,
                        cu_seqlens_q,
                        seqused_k,
                        block_table_gpu,
                        max_seqlen_q,
                        max_seqlen_k,
                        kv_cache,
                        device,
                        last_token_indices,
                    )
                }
            };

            // --- Merge logits in original request order ---
            // Allocate [num_reqs, vocab_size] and scatter decode/prefill logits.
            let row_bytes = vocab_size * decode_logits.dtype().size_bytes();
            let merged_owned = device
                .caching
                .alloc_tensor(&[num_reqs, vocab_size], decode_logits.dtype());
            let merged = merged_owned.as_gpu_tensor();

            // Copy decode logits into merged at their original positions.
            for (src_row, &orig_idx) in decode_indices.iter().enumerate() {
                let src = unsafe { decode_logits.raw_ptr().add(src_row * row_bytes) };
                let dst = unsafe { merged.raw_ptr().add(orig_idx * row_bytes) };
                unsafe { driver::memcpy_dtod_async(dst, src, row_bytes, device.compute_stream) }
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("mixed merge decode D2D: {e}"))
                    })?;
            }

            // Copy prefill logits into merged at their original positions.
            for (src_row, &orig_idx) in prefill_indices.iter().enumerate() {
                let src = unsafe { prefill_logits.raw_ptr().add(src_row * row_bytes) };
                let dst = unsafe { merged.raw_ptr().add(orig_idx * row_bytes) };
                unsafe { driver::memcpy_dtod_async(dst, src, row_bytes, device.compute_stream) }
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("mixed merge prefill D2D: {e}"))
                    })?;
            }

            // Invalidate graph metadata since we changed batch composition.
            self.last_graph_batch_size = None;
            self.graph_metadata_valid = false;

            // Drop the sub-logits so their memory returns to the caching allocator.
            // (decode_logits is a view into graph output — not owned. prefill_logits is
            // from forward_owned, also a view. merged_owned keeps the merged allocation.)
            let _ = decode_logits;
            let _ = prefill_logits;

            // Fall through to sampling with merged logits.
            let logits = merged;

            // GPU sampling for mixed prefill+decode merged logits.
            return Self::gpu_sample_and_finalize(
                &self.sampling_params_map,
                #[cfg(feature = "guided-decoding")]
                &mut self.grammar_states,
                &self.host_staging,
                &mut self.input_batch,
                &mut self.token_buffers,
                logits,
                prepared,
                device,
                all_greedy,
                vocab_size,
            );
        }

        // Check if any request needs logprobs or grammar (these require the full
        // GPU sampling pipeline instead of in-graph argmax).
        let any_needs_full_sampling = prepared.req_inputs.iter().any(|r| {
            self.sampling_params_map
                .get(&r.req_id)
                .is_some_and(|p| p.logprobs.is_some())
                || {
                    #[cfg(feature = "guided-decoding")]
                    {
                        self.grammar_states.contains_key(&r.req_id)
                    }
                    #[cfg(not(feature = "guided-decoding"))]
                    {
                        false
                    }
                }
        });

        if use_graph && all_greedy && !any_needs_full_sampling {
            // Fast path: CUDA graph with in-graph argmax. No separate sampling
            // kernel launch — argmax + D2D scatter are captured in the graph.
            let graph_bs = graph_bs.unwrap();
            let meta = &prepared.attn_meta;
            let staging = self.host_staging.as_ref();

            let replay_out = if self.graph_metadata_valid
                && self.last_graph_batch_size == Some(graph_bs)
            {
                // GPU-side metadata update: positions, slot_mapping, seqused_k
                // are incremented on GPU in a single kernel. Only block_table is
                // H2D-copied when blocks changed. input_ids were scattered by the
                // previous graph replay's in-graph argmax.
                let new_bt = if blocks_changed {
                    let bt =
                        unsafe { staging.unwrap().fill_block_table(&meta.block_ids, graph_bs) };
                    Some(bt)
                } else {
                    None
                };

                let runner = self.graph_runner.as_ref().unwrap();
                unsafe {
                    runner.replay_decode_fast(
                        graph_bs, None, // input_ids already scattered by previous graph
                        new_bt, block_size, device,
                    )
                }
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("graph replay_decode_fast: {e}"))
                })?
            } else if let Some(stg) = staging {
                // First step for this batch or batch composition changed:
                // full H2D of all metadata via pinned staging buffers.
                unsafe {
                    let ids = stg.input_ids.slice_mut::<u32>(graph_bs);
                    ids[..num_reqs].copy_from_slice(&prepared.flat_token_ids);
                    ids[num_reqs..].fill(0);

                    let pos = stg.positions.slice_mut::<u32>(graph_bs);
                    pos[..num_reqs].copy_from_slice(&prepared.flat_positions);
                    pos[num_reqs..].fill(0);

                    let cu_q = stg.cu_seqlens_q.slice_mut::<i32>(graph_bs + 1);
                    for (i, v) in cu_q[..=num_reqs].iter_mut().enumerate() {
                        *v = i as i32;
                    }
                    cu_q[(num_reqs + 1)..].fill(num_reqs as i32);

                    // seqused_k: per-sequence K lengths [graph_bs].
                    let sk = stg.seqused_k.slice_mut::<i32>(graph_bs);
                    for (i, &sl) in meta.seq_lens.iter().enumerate() {
                        sk[i] = sl as i32;
                    }
                    // Padded slots: use 1 (dummy seqs with 1 K token).
                    for s in &mut sk[num_reqs..] {
                        *s = 1;
                    }

                    let sm = stg.slot_mapping.slice_mut::<i64>(graph_bs);
                    for (i, slot) in sm[..num_reqs].iter_mut().enumerate() {
                        let abs_pos = meta.tokens_before[i];
                        let block_idx = abs_pos / block_size;
                        let offset = abs_pos % block_size;
                        let block_ids = &meta.block_ids[i];
                        if block_idx < block_ids.len() {
                            *slot = (block_ids[block_idx] * block_size + offset) as i64;
                        } else {
                            *slot = -1i64;
                        }
                    }
                    sm[num_reqs..].fill(-1i64);

                    let bt = stg.fill_block_table(&meta.block_ids, graph_bs);

                    let skip_input_ids =
                        self.last_graph_batch_size == Some(graph_bs) && num_reqs == graph_bs;

                    let runner = self.graph_runner.as_ref().unwrap();
                    runner.replay(
                        graph_bs,
                        stg.input_ids.slice::<u32>(graph_bs),
                        stg.positions.slice::<u32>(graph_bs),
                        stg.slot_mapping.slice::<i64>(graph_bs),
                        stg.cu_seqlens_q.slice::<i32>(graph_bs + 1),
                        stg.seqused_k.slice::<i32>(graph_bs),
                        bt,
                        device,
                        skip_input_ids,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("graph replay: {e}")))?
            } else {
                // Fallback: no pinned staging (shouldn't happen but safe).
                let mut input_ids = prepared.flat_token_ids.clone();
                input_ids.resize(graph_bs, 0);
                let mut positions = prepared.flat_positions.clone();
                positions.resize(graph_bs, 0);

                let mut cu_seqlens_q: Vec<i32> = (0..=num_reqs as i32).collect();
                for _ in num_reqs..graph_bs {
                    cu_seqlens_q.push(num_reqs as i32);
                }

                // seqused_k: per-sequence K lengths [graph_bs].
                let mut seqused_k: Vec<i32> = meta.seq_lens.iter().map(|&sl| sl as i32).collect();
                // Padded slots: use 1 (dummy seqs with 1 K token).
                seqused_k.resize(graph_bs, 1);

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

                let mut block_table = vec![0i32; graph_bs * GRAPH_MAX_BLOCKS_PER_SEQ];
                for (i, blocks) in meta.block_ids.iter().enumerate() {
                    for (j, &bid) in blocks.iter().enumerate() {
                        if j < GRAPH_MAX_BLOCKS_PER_SEQ {
                            block_table[i * GRAPH_MAX_BLOCKS_PER_SEQ + j] = bid as i32;
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
                        &seqused_k,
                        &block_table,
                        device,
                        skip_input_ids,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("graph replay: {e}")))?
            };

            // Record that graph buffers now have valid metadata for next step.
            self.last_graph_batch_size = Some(graph_bs);
            self.graph_metadata_valid = true;

            return Self::finalize_d2h_and_commit(
                &replay_out.token_ids,
                num_reqs,
                prepared,
                device,
                &self.host_staging,
                &mut self.input_batch,
                &mut self.token_buffers,
            );
        }

        // Non-greedy graph path or eager path: need separate sampling.
        let logits = if use_graph {
            // CUDA graph replay (non-greedy: in-graph argmax result is
            // discarded; we re-sample with temperature on the logits).
            let graph_bs = graph_bs.unwrap();
            let meta = &prepared.attn_meta;
            let staging = self.host_staging.as_ref();

            let replay_out = if self.graph_metadata_valid
                && self.last_graph_batch_size == Some(graph_bs)
            {
                // Fast path: GPU-side metadata update (same as greedy).
                // Only input_ids must be H2D'd (no in-graph argmax scatter for non-greedy).
                let new_bt = if blocks_changed {
                    let bt =
                        unsafe { staging.unwrap().fill_block_table(&meta.block_ids, graph_bs) };
                    Some(bt)
                } else {
                    None
                };

                // Must H2D input_ids since non-greedy doesn't use in-graph argmax scatter.
                let input_ids_slice = if let Some(stg) = staging {
                    unsafe {
                        let ids = stg.input_ids.slice_mut::<u32>(graph_bs);
                        ids[..num_reqs].copy_from_slice(&prepared.flat_token_ids);
                        ids[num_reqs..].fill(0);
                        stg.input_ids.slice::<u32>(graph_bs)
                    }
                } else {
                    let mut ids = prepared.flat_token_ids.clone();
                    ids.resize(graph_bs, 0);
                    ids.leak()
                };

                let runner = self.graph_runner.as_ref().unwrap();
                unsafe {
                    runner.replay_decode_fast(
                        graph_bs,
                        Some(input_ids_slice),
                        new_bt,
                        block_size,
                        device,
                    )
                }
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("graph replay_decode_fast: {e}"))
                })?
            } else if let Some(stg) = staging {
                // First step or batch composition changed: full H2D via pinned staging.
                unsafe {
                    let ids = stg.input_ids.slice_mut::<u32>(graph_bs);
                    ids[..num_reqs].copy_from_slice(&prepared.flat_token_ids);
                    ids[num_reqs..].fill(0);

                    let pos = stg.positions.slice_mut::<u32>(graph_bs);
                    pos[..num_reqs].copy_from_slice(&prepared.flat_positions);
                    pos[num_reqs..].fill(0);

                    let cu_q = stg.cu_seqlens_q.slice_mut::<i32>(graph_bs + 1);
                    for (i, v) in cu_q[..=num_reqs].iter_mut().enumerate() {
                        *v = i as i32;
                    }
                    cu_q[(num_reqs + 1)..].fill(num_reqs as i32);

                    // seqused_k: per-sequence K lengths [graph_bs].
                    let sk = stg.seqused_k.slice_mut::<i32>(graph_bs);
                    for (i, &sl) in meta.seq_lens.iter().enumerate() {
                        sk[i] = sl as i32;
                    }
                    for s in &mut sk[num_reqs..] {
                        *s = 1;
                    }

                    let sm = stg.slot_mapping.slice_mut::<i64>(graph_bs);
                    for (i, slot) in sm[..num_reqs].iter_mut().enumerate() {
                        let abs_pos = meta.tokens_before[i];
                        let block_idx = abs_pos / block_size;
                        let offset = abs_pos % block_size;
                        let block_ids = &meta.block_ids[i];
                        if block_idx < block_ids.len() {
                            *slot = (block_ids[block_idx] * block_size + offset) as i64;
                        } else {
                            *slot = -1i64;
                        }
                    }
                    sm[num_reqs..].fill(-1i64);

                    let bt = stg.fill_block_table(&meta.block_ids, graph_bs);

                    let runner = self.graph_runner.as_ref().unwrap();
                    runner.replay(
                        graph_bs,
                        stg.input_ids.slice::<u32>(graph_bs),
                        stg.positions.slice::<u32>(graph_bs),
                        stg.slot_mapping.slice::<i64>(graph_bs),
                        stg.cu_seqlens_q.slice::<i32>(graph_bs + 1),
                        stg.seqused_k.slice::<i32>(graph_bs),
                        bt,
                        device,
                        false,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("graph replay: {e}")))?
            } else {
                // Fallback: no pinned staging.
                let mut input_ids = prepared.flat_token_ids.clone();
                input_ids.resize(graph_bs, 0);
                let mut positions = prepared.flat_positions.clone();
                positions.resize(graph_bs, 0);

                let mut cu_seqlens_q: Vec<i32> = (0..=num_reqs as i32).collect();
                for _ in num_reqs..graph_bs {
                    cu_seqlens_q.push(num_reqs as i32);
                }

                // seqused_k: per-sequence K lengths [graph_bs].
                let mut seqused_k: Vec<i32> = meta.seq_lens.iter().map(|&sl| sl as i32).collect();
                // Padded slots: use 1 (dummy seqs with 1 K token).
                seqused_k.resize(graph_bs, 1);

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

                let mut block_table = vec![0i32; graph_bs * GRAPH_MAX_BLOCKS_PER_SEQ];
                for (i, blocks) in meta.block_ids.iter().enumerate() {
                    for (j, &bid) in blocks.iter().enumerate() {
                        if j < GRAPH_MAX_BLOCKS_PER_SEQ {
                            block_table[i * GRAPH_MAX_BLOCKS_PER_SEQ + j] = bid as i32;
                        }
                    }
                }

                let runner = self.graph_runner.as_ref().unwrap();
                unsafe {
                    runner.replay(
                        graph_bs,
                        &input_ids,
                        &positions,
                        &slot_mapping,
                        &cu_seqlens_q,
                        &seqused_k,
                        &block_table,
                        device,
                        false,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("graph replay: {e}")))?
            };

            // Track metadata validity for next step (works for non-greedy too).
            self.last_graph_batch_size = Some(graph_bs);
            self.graph_metadata_valid = true;

            // Slice logits to only the real requests (discard padded rows).
            if graph_bs > num_reqs {
                replay_out.logits.narrow_dim0(0, num_reqs)
            } else {
                replay_out.logits
            }
        } else {
            // Non-decode path: try prefill graph, fall back to eager.
            self.last_graph_batch_size = None;
            self.graph_metadata_valid = false;

            // Check if we can use a prefill graph: single request, fresh prefill
            // (tokens_before == 0 means q_len == seq_len, so the model uses contiguous
            // FA2 — not paged — which is safe to capture in a CUDA graph).
            let meta = &prepared.attn_meta;
            let use_prefill_graph = num_reqs == 1
                && meta.tokens_before[0] == 0
                && !any_needs_full_sampling
                && self
                    .prefill_graph_runner
                    .as_ref()
                    .and_then(|r| r.nearest_graph_size(total_tokens))
                    .is_some();

            if use_prefill_graph {
                let padded = self
                    .prefill_graph_runner
                    .as_ref()
                    .unwrap()
                    .nearest_graph_size(total_tokens)
                    .unwrap();

                // Build block_table padded to MAX_BLOCKS_PER_SEQ.
                let mut block_table = vec![0i32; GRAPH_MAX_BLOCKS_PER_SEQ];
                for (j, &bid) in meta.block_ids[0].iter().enumerate() {
                    if j < GRAPH_MAX_BLOCKS_PER_SEQ {
                        block_table[j] = bid as i32;
                    }
                }

                // Build slot_mapping for real tokens.
                let mut slot_mapping = Vec::with_capacity(total_tokens);
                let block_ids = &meta.block_ids[0];
                for t in 0..total_tokens {
                    let abs_pos = meta.tokens_before[0] + t;
                    let block_idx = abs_pos / block_size;
                    let offset = abs_pos % block_size;
                    if block_idx < block_ids.len() {
                        slot_mapping.push((block_ids[block_idx] * block_size + offset) as i64);
                    } else {
                        slot_mapping.push(-1i64);
                    }
                }

                let last_token_idx = (total_tokens - 1) as u32;

                let replay_out = unsafe {
                    self.prefill_graph_runner.as_ref().unwrap().replay(
                        padded,
                        &prepared.flat_token_ids,
                        &prepared.flat_positions,
                        &slot_mapping,
                        meta.seq_lens[0],
                        &block_table,
                        last_token_idx,
                        device,
                    )
                }
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("prefill graph replay: {e}"))
                })?;

                replay_out.logits
            } else {
                // Eager forward path (multi-request prefill or uncaptured size).
                // Caching allocator: no reset needed — tensors freed on drop.

                let gpu_input_ids = Self::h2d_u32(&prepared.flat_token_ids, device)?;
                let gpu_positions = Self::h2d_u32(&prepared.flat_positions, device)?;

                let (
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                ) = Self::build_attention_tensors(&prepared.attn_meta, block_size, device)?;

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
                    model.forward_owned(
                        gpu_input_ids,
                        gpu_positions,
                        slot_mapping,
                        cu_seqlens_q,
                        seqused_k,
                        block_table,
                        max_seqlen_q,
                        max_seqlen_k,
                        kv_cache,
                        device,
                        last_token_indices,
                    )
                }
            }
        };

        // GPU sampling: handles all cases — greedy, non-greedy, penalties,
        // grammar, logit_bias, logprobs — entirely on GPU. No CPU fallback.
        Self::gpu_sample_and_finalize(
            &self.sampling_params_map,
            #[cfg(feature = "guided-decoding")]
            &mut self.grammar_states,
            &self.host_staging,
            &mut self.input_batch,
            &mut self.token_buffers,
            logits,
            prepared,
            device,
            all_greedy,
            vocab_size,
        )
    }
} // end impl CudaWorker (execute_model_inner)

/// Compute KV cache budget matching Python vLLM's formula exactly:
///   requested = total_memory * gpu_memory_utilization
///   non_kv_cache = weights_and_overhead + peak_activations + 150 MiB
///   available_kv_bytes = requested - non_kv_cache
///
/// This is the exact logic used in `CudaWorker::determine_available_memory`.
pub fn compute_available_kv_bytes(
    total_memory: usize,
    weights_and_overhead: usize,
    peak_activation_bytes: usize,
    gpu_memory_utilization: f64,
) -> usize {
    let redundancy_buffer: usize = 150 * 1024 * 1024; // 150 MiB
    let non_kv_cache = weights_and_overhead + peak_activation_bytes + redundancy_buffer;
    let requested = (total_memory as f64 * gpu_memory_utilization) as usize;
    requested.saturating_sub(non_kv_cache)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    /// Verify KV cache budget matches Python vLLM's formula on real hardware.
    ///
    /// Python (gpu_worker.py):
    ///   requested = total_memory * gpu_memory_utilization
    ///   non_kv_cache = weights + peak_activations + non_torch + 150 MiB
    ///   available_kv = requested - non_kv_cache
    ///
    /// This test allocates known amounts of GPU memory, then calls
    /// `compute_available_kv_bytes` (the same formula CudaWorker uses)
    /// and verifies the result matches manual Python-style calculation.
    #[test]
    fn test_kv_cache_budget_matches_python_formula() {
        // L40S-like GPU: 46 GiB total.
        let total_memory: usize = 46 * 1024 * 1024 * 1024;

        // Simulate Qwen2.5-3B-like numbers.
        let model_weights: usize = 6 * 1024 * 1024 * 1024; // 6 GiB
        let peak_activations: usize = 3 * 1024 * 1024 * 1024; // 3 GiB
        let utilization = 0.9;

        let result =
            compute_available_kv_bytes(total_memory, model_weights, peak_activations, utilization);

        // Manually compute expected using Python's exact formula:
        //   requested = total * util
        //   non_kv = weights + peak + 150 MiB
        //   available = requested - non_kv
        let redundancy: usize = 150 * 1024 * 1024;
        let non_kv = model_weights + peak_activations + redundancy;
        let requested = (total_memory as f64 * utilization) as usize;
        let expected = requested.saturating_sub(non_kv);

        assert_eq!(
            result,
            expected,
            "KV budget {:.2} GiB != expected {:.2} GiB",
            result as f64 / 1_073_741_824.0,
            expected as f64 / 1_073_741_824.0,
        );

        // Sanity: ~32 GiB for KV, matching Python's "Available KV cache memory: 33.49 GiB"
        let result_gib = result as f64 / 1_073_741_824.0;
        assert!(
            result_gib > 30.0 && result_gib < 35.0,
            "Expected ~32 GiB KV budget for 46 GiB GPU with 3B model, got {:.1} GiB",
            result_gib
        );
    }

    /// Verify that utilization=0.5 gives less KV budget than 0.9.
    #[test]
    fn test_kv_cache_budget_respects_utilization() {
        let total: usize = 48 * 1024 * 1024 * 1024; // 48 GiB
        let weights: usize = 6 * 1024 * 1024 * 1024;
        let peak: usize = 3 * 1024 * 1024 * 1024;

        let budget_90 = compute_available_kv_bytes(total, weights, peak, 0.9);
        let budget_50 = compute_available_kv_bytes(total, weights, peak, 0.5);

        assert!(
            budget_50 < budget_90,
            "util=0.5 ({:.1} GiB) should give less KV than util=0.9 ({:.1} GiB)",
            budget_50 as f64 / 1_073_741_824.0,
            budget_90 as f64 / 1_073_741_824.0,
        );
    }

    /// Verify that when non_kv_cache exceeds requested memory, we get 0 (not underflow).
    #[test]
    fn test_kv_cache_budget_saturates_at_zero() {
        let total: usize = 8 * 1024 * 1024 * 1024; // 8 GiB total
        let weights: usize = 6 * 1024 * 1024 * 1024; // 6 GiB weights
        let peak: usize = 3 * 1024 * 1024 * 1024; // 3 GiB peak
        // non_kv = 6 + 3 + 0.15 = 9.15 GiB > requested = 8 * 0.9 = 7.2 GiB

        let budget = compute_available_kv_bytes(total, weights, peak, 0.9);
        assert_eq!(budget, 0, "should saturate at 0, not underflow");
    }
}
