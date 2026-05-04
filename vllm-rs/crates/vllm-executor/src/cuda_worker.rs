// SPDX-License-Identifier: Apache-2.0
//! `CudaWorker`: a `Worker` implementation using the vllm-cuda backend.
//!
//! Purpose-built GPU runtime with `GpuTensor`/`GpuDevice`/`ScratchArena`
//! for zero-allocation inference. Uses the same paged FlashAttention-2 kernels,
//! but through raw FFI instead of CustomOps.
//!
//! This worker is gated behind the `cuda-backend` feature flag.

// Force the linker to keep `ferrite_models` — its `#[forward]`
// modules register with `inventory::submit!` for auto-discovery by
// `ferrite_forward::try_load`. Without this, the linker gc's the
// crate (no direct symbol references after the Phase B collapse)
// and the inventory comes up empty.
extern crate ferrite_models as _;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tracing::info;
use vllm_common::SamplingParams;
use vllm_common::engine_io::EmbeddingData;
use vllm_config::CudaGraphMode;
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_cuda::OwnedTensor;
use vllm_cuda::cpu_gpu_buf::PinnedBuf;
use vllm_cuda::device::GpuDevice;
use vllm_cuda::driver;
use vllm_cuda::dtype::DType as GpuDType;
use vllm_cuda::graph::{CudaGraphRunner, PrefillGraphRunner};
use vllm_cuda::graph_piece::{GraphPieceType, PiecewiseGraphRunner};
use vllm_cuda::kv_cache::KvCachePool;
use vllm_cuda::logits_processor::{
    AllowedTokenIdsProcessor, BadWordsProcessor, BatchUpdate, GrammarMaskProcessor,
    LogitBiasProcessor, LogitsProcessor, LogitsProcessorPipeline, MinTokensProcessor,
    PenaltiesProcessor, SealPadProcessor,
};
use vllm_cuda::quant;
use vllm_cuda::tensor::{GpuTensor, TensorView};
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
    /// CUDA graph mode: controls piecewise vs monolithic graph capture.
    pub cuda_graph_mode: CudaGraphMode,
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
    /// Tensor parallelism rank (0 = single GPU / rank 0).
    pub tp_rank: usize,
    /// Tensor parallelism world size (1 = no TP).
    pub tp_world_size: usize,
    /// Pipeline parallelism rank (0 = first stage).
    pub pp_rank: usize,
    /// Pipeline parallelism size (1 = no PP).
    pub pp_size: usize,
    /// Optional GGUF file name for HF Hub download (e.g. "model-Q4_K_M.gguf").
    pub gguf_file: Option<String>,
    /// Optional LoRA adapter path (local directory or HF repo ID).
    pub lora_adapter: Option<String>,
    /// KV cache data type: "auto" (use model dtype) or "fp8_e4m3".
    pub kv_cache_dtype: String,
    /// Compute KV scales dynamically from the first forward pass.
    pub calculate_kv_scales: bool,
    /// EOS token IDs for seal-pad processor (from model config).
    pub eos_token_ids: Vec<u32>,
    /// Runtime `max_model_len` (CLI `--max-model-len`). `None` falls
    /// back to `hf_config.max_position_embeddings` at load time.
    /// Threaded into ferrite's `try_load` so Phi-3 LongRoPE can
    /// decide `use_long_rope = max_model_len > original_max_pos`
    /// the same way Python vLLM does at init time.
    pub max_model_len: Option<usize>,
}

/// Per-image MRoPE metadata in seq space, used by
/// [`CudaWorker::build_mrope_positions_2d`] to walk the entire seq for
/// each MM-bearing req (including the cached-prefix portion) so the
/// cursor lands at the same position the encoder used to encode KV.
///
/// Tuple is `(seq_offset, length, grid_t, grid_h_merged, grid_w_merged)`
/// where `seq_offset` is relative to the req's full sequence start (NOT
/// batch start) — the same coordinate system as `PlaceholderRange.offset`.
type SeqMmInfo = (u32, u32, u32, u32, u32);

// ---------------------------------------------------------------------------
// Model enum (dispatches to LLaMA / Qwen2 / Gemma2)
// ---------------------------------------------------------------------------

/// Per-forward multimodal inputs threaded into `CudaModel::forward` /
/// `hidden_states`. Built once per step in `execute_model_inner` from the
/// active batch's cached `MultimodalData`, consumed only by the
/// `Self::Ferrite` arm — every other (text-only) arm ignores it.
///
/// `mm_embeds` is the projected vision-encoder output `[total_mm_tokens,
/// hidden]` produced by `MultimodalForward::vision_forward`;
/// `embed_patches` names where each image's slice lands in the flat input-
/// id sequence (token-space). The Ferrite arm copies both into
/// `ForwardCtx`, where `Instruction::Embed::eval` D2D-splices each patch
/// into the gather output. Empty / `None` is the text-only no-op path —
/// byte-identical to before MM landed.
pub struct MmForwardInputs<'a> {
    pub mm_embeds: TensorView<'a>,
    pub embed_patches: &'a [ferrite_forward::EmbedPatch],
}

/// A dense-bf16 Llama loaded through the new ferrite-forward
/// compiler. Holds the specialized `Weights` enum (one variant per
/// compiled config) + the `RotaryCache` the forward needs. Accessors
/// (num_layers / num_kv_heads / head_dim / vocab_size / hidden_size)
/// come directly from `Weights`'s methods — baked in at macro-
/// expansion time from each model's config.json.
///
/// Only populated when `VLLM_FERRITE=1` at load time; otherwise a
/// dense-bf16 Llama routes to `CudaModel::LlamaSolver` as before.
/// Any ferrite-compiled model, loaded through
/// [`ferrite_forward::try_load`]. The trait object erases the arch;
/// accessors + forward/backbone go through the `FerriteWeights`
/// vtable. One struct covers every registered arch (llama, qwen2,
/// qwen3, gemma2, granite, and any future arch that lands a
/// `#[forward]` module) — adding a new arch requires zero lines
/// here.
pub struct FerriteModel {
    pub weights: Box<dyn ferrite_forward::FerriteWeights>,
    /// Vision-encoder handle for multimodal arches. `Some(_)` only
    /// when the loaded checkpoint carries `visual.*` tensors AND the
    /// arch has submitted a `FerriteMmRegistration` row claiming the
    /// HF arch string. cuda_worker calls
    /// `mm.vision_forward(pixel_batches, placeholders, device)` when
    /// the request batch carries `mm_data`; the result threads into
    /// `ForwardCtx.mm_embeds` / `embed_patches` for the
    /// `Instruction::Embed::eval` splice. `None` for every text-only
    /// arch — the splice arm sees `embed_patches.is_empty()` and
    /// degenerates to plain `embedding_gather_masked`.
    pub mm: Option<Box<dyn ferrite_forward::MultimodalForward>>,
    /// NCCL communicator the worker injects via `set_tp_group` after
    /// construction. Threaded into `ForwardCtx::tp_group` on every
    /// forward call — the universal `Instruction::AllReduce` arm
    /// (gated under `feature = "nccl"`) reads it and dispatches into
    /// `NcclGroup::all_reduce_inplace`. `None` at tp=1 (the lowering
    /// pass emits zero AllReduce rows, so the field is never read).
    #[cfg(feature = "nccl")]
    pub tp_group: Option<std::sync::Arc<vllm_cuda::nccl::NcclGroup>>,
    /// Tensor-parallel world size baked into the loaded `Weights`
    /// variant — used to derive the per-rank `num_kv_heads` for KV
    /// cache allocation. `FerriteWeights::num_key_value_heads()` is
    /// the *unsharded* config value (sharded values flow through
    /// `CanonicalParams`); the executor must divide by `tp_world_size`
    /// here to match hand-written `self_attn.num_kv_heads` (which is
    /// already sharded) and Python's per-rank `num_kv_heads`. Without
    /// this, `KvCachePool` is sized with the unsharded head count and
    /// the per-block stride disagrees with the kernel-side per-rank
    /// `NUM_KV_HEADS` from `CanonicalParams`, producing decode garbage
    /// at tp>1.
    pub tp_world_size: usize,
}

/// Supported model architectures in the vllm-cuda backend.
enum CudaModel {
    Llama(vllm_cuda::model::llama::LlamaForCausalLM),
    Qwen2(vllm_cuda::model::qwen2::Qwen2ForCausalLM),
    Gemma2(vllm_cuda::model::gemma2::Gemma2ForCausalLM),
    /// Any ferrite-compiled arch. Populated by
    /// `ferrite_forward::try_load(gw, stream, hf_arch_name)` —
    /// one variant covers every registered `#[forward]` module
    /// (llama / qwen2 / qwen3 / gemma2 / granite / …). Boxed
    /// because the per-arch `Weights` enum has one variant per
    /// compiled model config and can be large.
    Ferrite(Box<FerriteModel>),
    Gemma3(vllm_cuda::model::gemma3::Gemma3ForCausalLM),
    Mixtral(vllm_cuda::model::mixtral::MixtralForCausalLM),
    Qwen2Moe(vllm_cuda::model::qwen2_moe::Qwen2MoeForCausalLM),
    Qwen3Moe(vllm_cuda::model::qwen3_moe::Qwen3MoeForCausalLM),
    CommandR(vllm_cuda::model::commandr::CommandRForCausalLM),
    Qwen3Next(vllm_cuda::model::qwen3_next::Qwen3NextForCausalLM),
    DeepSeekV2(vllm_cuda::model::deepseek_v2::DeepSeekV2ForCausalLM),
}

impl CudaModel {
    /// Hybrid arches that need an auxiliary recurrent-state pool
    /// alongside the paged KV cache (Mamba-style conv1d / SSM state).
    /// True for the hand-written `Qwen3Next` variant and for ferrite
    /// loads whose `arch_name() == "qwen3_next"`. Drives the dispatch
    /// site that builds GDN tensors before forward.
    fn is_qwen3_next(&self) -> bool {
        match self {
            Self::Qwen3Next(_) => true,
            Self::Ferrite(m) => m.weights.arch_name() == "qwen3_next",
            _ => false,
        }
    }

    fn num_layers(&self) -> usize {
        match self {
            Self::Llama(m) => m.model.layers.len(),
            Self::Qwen2(m) => m.0.model.layers.len(),
            Self::Gemma2(m) => m.model.layers.len(),
            Self::Ferrite(m) => m.weights.num_hidden_layers() as usize,
            Self::Gemma3(m) => m.model.layers.len(),
            Self::Mixtral(m) => m.model.layers.len(),
            Self::Qwen2Moe(m) => m.model.layers.len(),
            Self::Qwen3Moe(m) => m.model.layers.len(),
            Self::CommandR(m) => m.model.layers.len(),
            // Only full attention layers need KV cache.
            Self::Qwen3Next(m) => m.num_kv_layers(),
            Self::DeepSeekV2(m) => m.model.layers.len(),
        }
    }

    /// Whether this model is a ferrite-compiled encoder (no lm_head Gemm,
    /// returns hidden states from `forward_backbone`). Encoders skip CUDA
    /// graph capture and use the fixed-estimate profiling path. Today only
    /// modernbert qualifies; add new arch names here as they land.
    fn is_ferrite_encoder(&self) -> bool {
        match self {
            Self::Ferrite(m) => matches!(m.weights.arch_name(), "modernbert"),
            _ => false,
        }
    }

    fn num_kv_heads(&self) -> usize {
        match self {
            Self::Llama(m) => m.model.layers[0].self_attn.num_kv_heads,
            Self::Qwen2(m) => m.0.model.layers[0].self_attn.num_kv_heads,
            Self::Gemma2(m) => m.model.layers[0].self_attn.num_kv_heads,
            Self::Ferrite(m) => {
                // FerriteWeights reports the unsharded config value;
                // divide by tp_world_size so KvCachePool gets the
                // per-rank head count, matching hand-written
                // `self_attn.num_kv_heads` and Python's
                // `max(1, total // tp_size)`.
                let total = m.weights.num_key_value_heads() as usize;
                let tp = m.tp_world_size.max(1);
                (total / tp).max(1)
            }
            Self::Gemma3(m) => m.model.layers[0].self_attn.num_kv_heads,
            Self::Mixtral(m) => m.model.layers[0].self_attn.num_kv_heads,
            Self::Qwen2Moe(m) => m.model.layers[0].self_attn.num_kv_heads,
            Self::Qwen3Moe(m) => m.model.layers[0].self_attn.num_kv_heads,
            Self::CommandR(m) => m.model.layers[0].self_attn.inner.num_kv_heads,
            Self::Qwen3Next(m) => m.num_kv_heads(),
            Self::DeepSeekV2(m) => m.model.layers[0].self_attn.num_heads,
        }
    }

    fn head_dim(&self) -> usize {
        match self {
            Self::Llama(m) => m.model.layers[0].self_attn.head_dim,
            Self::Qwen2(m) => m.0.model.layers[0].self_attn.head_dim,
            Self::Gemma2(m) => m.model.layers[0].self_attn.head_dim,
            Self::Ferrite(m) => m.weights.head_dim() as usize,
            Self::Gemma3(m) => m.model.layers[0].self_attn.head_dim,
            Self::Mixtral(m) => m.model.layers[0].self_attn.head_dim,
            Self::Qwen2Moe(m) => m.model.layers[0].self_attn.head_dim,
            Self::Qwen3Moe(m) => m.model.layers[0].self_attn.head_dim,
            Self::CommandR(m) => m.model.layers[0].self_attn.inner.head_dim,
            Self::Qwen3Next(m) => m.head_dim(),
            Self::DeepSeekV2(m) => m.model.layers[0].self_attn.qk_head_dim,
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Llama(m) => m.lm_head.out_features(),
            Self::Qwen2(m) => m.0.lm_head.out_features(),
            Self::Gemma2(m) => m.lm_head.out_features(),
            Self::Ferrite(m) => m.weights.vocab_size() as usize,
            Self::Gemma3(m) => m.lm_head.out_features(),
            Self::Mixtral(m) => m.lm_head.out_features(),
            Self::Qwen2Moe(m) => m.lm_head.out_features(),
            Self::Qwen3Moe(m) => m.lm_head.out_features(),
            Self::CommandR(m) => m.lm_head.out_features(),
            Self::Qwen3Next(m) => m.lm_head.out_features(),
            Self::DeepSeekV2(m) => m.lm_head.out_features(),
        }
    }

    /// Whether this model uses Mixture-of-Experts layers.
    /// MoE models generate too many CUDA graph nodes for monolithic capture
    /// (router + N expert GEMMs + shared experts per layer) and exceed the
    /// CUDA driver's undocumented node limit. Piecewise capture is required.
    fn is_moe(&self) -> bool {
        matches!(
            self,
            Self::Mixtral(_) | Self::Qwen2Moe(_) | Self::Qwen3Moe(_) | Self::DeepSeekV2(_)
        )
    }

    fn hidden_size(&self) -> usize {
        match self {
            Self::Llama(m) => m.lm_head.in_features(),
            Self::Qwen2(m) => m.0.lm_head.in_features(),
            Self::Gemma2(m) => m.lm_head.in_features(),
            Self::Ferrite(m) => m.weights.hidden_size() as usize,
            Self::Gemma3(m) => m.lm_head.in_features(),
            Self::Mixtral(m) => m.lm_head.in_features(),
            Self::Qwen2Moe(m) => m.lm_head.in_features(),
            Self::Qwen3Moe(m) => m.lm_head.in_features(),
            Self::CommandR(m) => m.lm_head.in_features(),
            Self::Qwen3Next(m) => m.lm_head.in_features(),
            Self::DeepSeekV2(m) => m.lm_head.in_features(),
        }
    }

    /// Inject NCCL process group into all model layers for TP.
    #[cfg(feature = "nccl")]
    fn set_tp_group(&mut self, group: std::sync::Arc<vllm_cuda::nccl::NcclGroup>) {
        match self {
            Self::Llama(m) => m.set_tp_group(group),
            Self::Qwen2(m) => m.0.set_tp_group(group),
            Self::Gemma2(m) => m.set_tp_group(group),
            Self::Gemma3(m) => m.set_tp_group(group),
            Self::Mixtral(m) => m.set_tp_group(group),
            Self::Qwen2Moe(m) => m.set_tp_group(group),
            Self::Qwen3Moe(m) => m.set_tp_group(group),
            Self::CommandR(_) => {}  // TP not yet supported
            Self::Qwen3Next(_) => {} // TP not yet supported
            Self::DeepSeekV2(m) => m.set_tp_group(group),
            Self::Ferrite(m) => m.tp_group = Some(group),
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
        input_ids: TensorView<'_>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
        mm_inputs: Option<&MmForwardInputs<'_>>,
    ) -> vllm_cuda::OwnedTensor {
        match self {
            Self::Llama(m) => unsafe {
                m.model.forward(
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
                m.0.model.forward(
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
                m.model.forward(
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
            // Ferrite arches all share this shape: build `ForwardCtx`
            // from the argument bag, dispatch through the
            // `FerriteWeights` trait's `forward_backbone` vtable. The
            // trait impl inside each arch's compiled module routes
            // to the per-arch `forward_backbone(&weights, &ctx, ...)`.
            Self::Ferrite(m) => unsafe {
                let num_tokens = input_ids.dim(0) as u64;
                let (mm_embeds, embed_patches) = match mm_inputs {
                    Some(mm) => (Some(mm.mm_embeds), mm.embed_patches),
                    None => (None, &[][..]),
                };
                let ctx = ferrite_forward::ForwardCtx {
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    mm_embeds,
                    embed_patches,
                    // GDN state is Qwen3-Next-only and threaded
                    // through a dedicated dispatch path; the
                    // ferrite forward dispatcher today doesn't run
                    // hybrid arches so both fields are None here.
                    gdn_state: None,
                    gdn_state_indices: None,
                    #[cfg(feature = "nccl")]
                    tp_group: m.tp_group.as_ref(),
                };
                m.weights.forward_backbone(&ctx, device, num_tokens)
            },
            Self::Gemma3(m) => unsafe {
                m.model.forward(
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
            Self::Mixtral(m) => unsafe {
                m.model.forward(
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
            Self::Qwen2Moe(m) => unsafe {
                m.model.forward(
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
            Self::Qwen3Moe(m) => unsafe {
                m.model.forward(
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
            Self::CommandR(m) => unsafe {
                m.model.forward(
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
            Self::Qwen3Next(_) => {
                panic!("Qwen3Next: use forward with GDN context");
            }
            Self::DeepSeekV2(m) => unsafe {
                m.model.forward(
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

    /// Forward using caching allocator (zero D2D copies between layers).
    /// Returns an `OwnedTensor` whose drop frees the logits allocation.
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward(
        &self,
        input_ids: TensorView<'_>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
        last_token_indices: Option<TensorView<'_>>,
        mm_inputs: Option<&MmForwardInputs<'_>>,
    ) -> vllm_cuda::OwnedTensor {
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
            // Every ferrite-compiled arch routes through the
            // `FerriteWeights` trait's `forward` vtable. The per-arch
            // macro-emitted impl delegates to that arch's specialized
            // `forward(&weights, &ctx, device, num_tokens)`.
            //
            // Selective last-token gather for prefill: the ferrite
            // DSL currently ends with `logits = gemm(.., lm_head)`
            // which runs lm_head over ALL num_tokens rows — so we
            // gather AFTER the matmul (correct, wasteful). A future
            // ferrite-level optimization would expose a gather op
            // in the DSL so the user can place it before lm_head.
            Self::Ferrite(m) => unsafe {
                // Encoder arches (modernbert) have no lm_head — both `forward`
                // and `forward_backbone` return hidden states. Routing those
                // through the logit-gather path below would mis-shape the
                // gather. Pooling callers use `hidden_states()` instead; this
                // arm is the logit-generation path and must hard-fail for
                // encoders.
                if self.is_ferrite_encoder() {
                    panic!(
                        "{}: encoder model does not support logit generation; use hidden_states()",
                        m.weights.arch_name()
                    );
                }
                let num_tokens = input_ids.dim(0) as u64;
                let (mm_embeds, embed_patches) = match mm_inputs {
                    Some(mm) => (Some(mm.mm_embeds), mm.embed_patches),
                    None => (None, &[][..]),
                };
                let ctx = ferrite_forward::ForwardCtx {
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    mm_embeds,
                    embed_patches,
                    // See companion `Self::Ferrite` arm above —
                    // hybrid GDN dispatch is not yet wired through
                    // this path, so both fields are None.
                    gdn_state: None,
                    gdn_state_indices: None,
                    #[cfg(feature = "nccl")]
                    tp_group: m.tp_group.as_ref(),
                };
                let logits = m.weights.forward(&ctx, device, num_tokens);
                match last_token_indices {
                    Some(idx) if idx.dim(0) < num_tokens as usize => {
                        vllm_cuda::kernels::embedding_gather(
                            logits.as_gpu_tensor(),
                            *idx,
                            &mut device.caching,
                            device.compute_stream,
                        )
                    }
                    _ => logits,
                }
            },
            Self::Gemma3(m) => unsafe {
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
            Self::Mixtral(m) => unsafe {
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
            Self::Qwen2Moe(m) => unsafe {
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
            Self::Qwen3Moe(m) => unsafe {
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
            Self::CommandR(m) => unsafe {
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
            Self::Qwen3Next(_) => {
                panic!("Qwen3Next: use forward_qwen3_next directly");
            }
            Self::DeepSeekV2(m) => unsafe {
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
        }
    }

    /// Forward pass for Qwen3Next with GDN context.
    ///
    /// Routes to either the hand-written `Qwen3Next::forward` (legacy
    /// path, single `vllm-cuda` model) or to the ferrite forward
    /// vtable with `ForwardCtx::{gdn_state, gdn_state_indices}`
    /// populated. Both paths consume the same per-request state
    /// pool + state-indices tensors built by `build_gdn_tensors`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward_qwen3_next(
        &self,
        input_ids: TensorView<'_>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        gdn_state_pool: &vllm_cuda::model::qwen3_next::GdnStatePool,
        gdn_state_indices: TensorView<'_>,
        gdn_cu_seqlens: TensorView<'_>,
        num_seqs: usize,
        device: &mut GpuDevice,
        last_token_indices: Option<TensorView<'_>>,
    ) -> vllm_cuda::OwnedTensor {
        match self {
            Self::Qwen3Next(m) => unsafe {
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
                    gdn_state_pool,
                    gdn_state_indices,
                    gdn_cu_seqlens,
                    num_seqs,
                    device,
                    last_token_indices,
                )
            },
            Self::Ferrite(m) => unsafe {
                let _ = (gdn_cu_seqlens, num_seqs);
                let num_tokens = input_ids.dim(0) as u64;
                let ctx = ferrite_forward::ForwardCtx {
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    mm_embeds: None,
                    embed_patches: &[],
                    gdn_state: Some(gdn_state_pool),
                    gdn_state_indices: Some(gdn_state_indices),
                    #[cfg(feature = "nccl")]
                    tp_group: m.tp_group.as_ref(),
                };
                let logits = m.weights.forward(&ctx, device, num_tokens);
                match last_token_indices {
                    Some(idx) if idx.dim(0) < num_tokens as usize => {
                        vllm_cuda::kernels::embedding_gather(
                            logits.as_gpu_tensor(),
                            *idx,
                            &mut device.caching,
                            device.compute_stream,
                        )
                    }
                    _ => logits,
                }
            },
            _ => panic!("forward_qwen3_next called on non-Qwen3Next model"),
        }
    }

    /// PP-aware forward: routes to the model's forward_pp method.
    /// Returns ForwardOutput::Logits on last stage, ForwardOutput::Intermediate otherwise.
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward_pp(
        &self,
        input_ids: Option<TensorView<'_>>,
        intermediate: Option<(vllm_cuda::OwnedTensor, vllm_cuda::OwnedTensor)>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
        last_token_indices: Option<TensorView<'_>>,
    ) -> vllm_cuda::model::llama::ForwardOutput {
        match self {
            Self::Llama(m) => unsafe {
                m.forward_pp(
                    input_ids,
                    intermediate,
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
                m.forward_pp(
                    input_ids,
                    intermediate,
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
                m.forward_pp(
                    input_ids,
                    intermediate,
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
                m.forward_pp(
                    input_ids,
                    intermediate,
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
            _ => panic!(
                "Pipeline parallelism not yet supported for {:?}",
                std::mem::discriminant(self)
            ),
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

fn mixtral_config_from_hf(
    hf: &HfModelConfig,
) -> ExecutorResult<vllm_cuda::model::mixtral::MixtralConfig> {
    let hidden_size = hf
        .hidden_size
        .ok_or_else(|| ExecutorError::WorkerInit("missing hidden_size".into()))?;
    let num_attention_heads = hf
        .num_attention_heads
        .ok_or_else(|| ExecutorError::WorkerInit("missing num_attention_heads".into()))?;
    let num_kv_heads = hf.num_key_value_heads.unwrap_or(num_attention_heads);
    let head_dim = hf.head_dim.unwrap_or(hidden_size / num_attention_heads);
    let num_local_experts = hf
        .extra
        .get("num_local_experts")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ExecutorError::WorkerInit("missing num_local_experts".into()))?
        as usize;
    let num_experts_per_tok = hf
        .extra
        .get("num_experts_per_tok")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ExecutorError::WorkerInit("missing num_experts_per_tok".into()))?
        as usize;

    Ok(vllm_cuda::model::mixtral::MixtralConfig {
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        num_hidden_layers: hf.num_hidden_layers.unwrap_or(32),
        intermediate_size: hf.intermediate_size.unwrap_or(hidden_size * 4),
        vocab_size: hf.vocab_size.unwrap_or(32000),
        max_position_embeddings: hf.max_position_embeddings.unwrap_or(4096),
        rms_norm_eps: hf.rms_norm_eps.unwrap_or(1e-5) as f32,
        rope_theta: hf.rope_theta.unwrap_or(1_000_000.0),
        head_dim,
        tie_word_embeddings: hf.tie_word_embeddings.unwrap_or(false),
        num_local_experts,
        num_experts_per_tok,
    })
}

fn qwen2_moe_config_from_hf(
    hf: &HfModelConfig,
) -> ExecutorResult<vllm_cuda::model::qwen2_moe::Qwen2MoeConfig> {
    let hidden_size = hf
        .hidden_size
        .ok_or_else(|| ExecutorError::WorkerInit("missing hidden_size".into()))?;
    let num_attention_heads = hf
        .num_attention_heads
        .ok_or_else(|| ExecutorError::WorkerInit("missing num_attention_heads".into()))?;
    let num_kv_heads = hf.num_key_value_heads.unwrap_or(num_attention_heads);
    let head_dim = hf.head_dim.unwrap_or(hidden_size / num_attention_heads);
    let num_experts = hf
        .extra
        .get("num_experts")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ExecutorError::WorkerInit("missing num_experts".into()))?
        as usize;
    let num_experts_per_tok = hf
        .extra
        .get("num_experts_per_tok")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ExecutorError::WorkerInit("missing num_experts_per_tok".into()))?
        as usize;
    let moe_intermediate_size = hf
        .extra
        .get("moe_intermediate_size")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ExecutorError::WorkerInit("missing moe_intermediate_size".into()))?
        as usize;
    let shared_expert_intermediate_size = hf
        .extra
        .get("shared_expert_intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let mlp_only_layers: Vec<usize> = hf
        .extra
        .get("decoder_sparse_step")
        .and_then(|v| v.as_u64())
        .map(|step| {
            // decoder_sparse_step=N means every N-th layer is MoE, others are dense.
            // mlp_only_layers = layers NOT divisible by step.
            let num_layers = hf.num_hidden_layers.unwrap_or(24);
            (0..num_layers).filter(|i| i % step as usize != 0).collect()
        })
        .or_else(|| {
            // Or explicit mlp_only_layers array.
            hf.extra.get("mlp_only_layers").and_then(|v| {
                v.as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_u64().map(|n| n as usize))
                        .collect()
                })
            })
        })
        .unwrap_or_default();

    Ok(vllm_cuda::model::qwen2_moe::Qwen2MoeConfig {
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        num_hidden_layers: hf.num_hidden_layers.unwrap_or(24),
        intermediate_size: hf.intermediate_size.unwrap_or(hidden_size * 4),
        moe_intermediate_size,
        shared_expert_intermediate_size,
        vocab_size: hf.vocab_size.unwrap_or(151936),
        max_position_embeddings: hf.max_position_embeddings.unwrap_or(32768),
        rms_norm_eps: hf.rms_norm_eps.unwrap_or(1e-6) as f32,
        rope_theta: hf.rope_theta.unwrap_or(1_000_000.0),
        head_dim,
        tie_word_embeddings: hf.tie_word_embeddings.unwrap_or(false),
        num_experts,
        num_experts_per_tok,
        mlp_only_layers,
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

    // Read layer_types from HF config if available, otherwise fall back to
    // Gemma2's default alternating pattern (even = sliding, odd = full).
    let layer_is_sliding: Vec<bool> = match hf.extra.get("layer_types").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .map(|v| v.as_str() == Some("sliding_attention"))
            .collect(),
        None => (0..num_hidden_layers).map(|i| i % 2 == 0).collect(),
    };

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

fn commandr_config_from_hf(
    hf: &HfModelConfig,
) -> ExecutorResult<vllm_cuda::model::commandr::CommandRConfig> {
    let hidden_size = hf
        .hidden_size
        .ok_or_else(|| ExecutorError::WorkerInit("missing hidden_size".into()))?;
    let num_attention_heads = hf
        .num_attention_heads
        .ok_or_else(|| ExecutorError::WorkerInit("missing num_attention_heads".into()))?;
    let num_kv_heads = hf.num_key_value_heads.unwrap_or(num_attention_heads);
    let head_dim = hf.head_dim.unwrap_or(hidden_size / num_attention_heads);

    let logit_scale = hf
        .extra
        .get("logit_scale")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let use_qk_norm = hf
        .extra
        .get("use_qk_norm")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let layer_norm_eps = hf
        .extra
        .get("layer_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;

    Ok(vllm_cuda::model::commandr::CommandRConfig {
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        num_hidden_layers: hf.num_hidden_layers.unwrap_or(32),
        intermediate_size: hf.intermediate_size.unwrap_or(hidden_size * 4),
        vocab_size: hf.vocab_size.unwrap_or(256000),
        max_position_embeddings: hf.max_position_embeddings.unwrap_or(8192),
        layer_norm_eps,
        rope_theta: hf.rope_theta.unwrap_or(10000.0),
        head_dim,
        logit_scale,
        use_qk_norm,
        tie_word_embeddings: hf.tie_word_embeddings.unwrap_or(true),
    })
}

fn qwen3_next_config_from_hf(
    hf: &HfModelConfig,
) -> ExecutorResult<vllm_cuda::model::qwen3_next::Qwen3NextConfig> {
    let hidden_size = hf
        .hidden_size
        .ok_or_else(|| ExecutorError::WorkerInit("missing hidden_size".into()))?;
    let num_attention_heads = hf
        .num_attention_heads
        .ok_or_else(|| ExecutorError::WorkerInit("missing num_attention_heads".into()))?;
    let num_kv_heads = hf.num_key_value_heads.unwrap_or(num_attention_heads);
    let num_hidden_layers = hf.num_hidden_layers.unwrap_or(28);
    let head_dim = hf.head_dim.unwrap_or(hidden_size / num_attention_heads);

    let partial_rotary_factor = hf
        .extra
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.25);
    let attn_output_gate = hf
        .extra
        .get("attn_output_gate")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // GDN fields
    let linear_conv_kernel_dim = hf
        .extra
        .get("linear_conv_kernel_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(4) as usize;
    let linear_key_head_dim = hf
        .extra
        .get("linear_key_head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(64) as usize;
    let linear_value_head_dim = hf
        .extra
        .get("linear_value_head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;
    let linear_num_key_heads = hf
        .extra
        .get("linear_num_key_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(4) as usize;
    let linear_num_value_heads = hf
        .extra
        .get("linear_num_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(4) as usize;

    // MoE fields
    let num_experts = hf
        .extra
        .get("num_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let num_experts_per_tok = hf
        .extra
        .get("num_experts_per_tok")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let moe_intermediate_size = hf
        .extra
        .get("moe_intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let shared_expert_intermediate_size = hf
        .extra
        .get("shared_expert_intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let norm_topk_prob = hf
        .extra
        .get("norm_topk_prob")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let decoder_sparse_step = hf
        .extra
        .get("decoder_sparse_step")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    let mlp_only_layers: Vec<usize> = hf
        .extra
        .get("mlp_only_layers")
        .and_then(|v| {
            v.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect()
            })
        })
        .unwrap_or_default();

    // Layer types: ["full_attention", "linear_attention", ...]
    let layer_types: Vec<String> = hf
        .extra
        .get("layer_types")
        .and_then(|v| {
            v.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
        })
        .unwrap_or_else(|| {
            // Default: every 4th layer is full attention.
            (0..num_hidden_layers)
                .map(|i| {
                    if i % 4 == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect()
        });

    Ok(vllm_cuda::model::qwen3_next::Qwen3NextConfig {
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        num_hidden_layers,
        intermediate_size: hf.intermediate_size.unwrap_or(hidden_size * 4),
        vocab_size: hf.vocab_size.unwrap_or(151936),
        max_position_embeddings: hf.max_position_embeddings.unwrap_or(32768),
        rms_norm_eps: hf.rms_norm_eps.unwrap_or(1e-6) as f32,
        rope_theta: hf.rope_theta.unwrap_or(1_000_000.0),
        head_dim,
        tie_word_embeddings: hf.tie_word_embeddings.unwrap_or(false),
        partial_rotary_factor,
        attn_output_gate,
        linear_conv_kernel_dim,
        linear_key_head_dim,
        linear_value_head_dim,
        linear_num_key_heads,
        linear_num_value_heads,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        shared_expert_intermediate_size,
        norm_topk_prob,
        decoder_sparse_step,
        mlp_only_layers,
        layer_types,
        layer_scale: hf
            .extra
            .get("layer_scale")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    })
}

fn deepseek_v2_config_from_hf(
    hf: &HfModelConfig,
) -> ExecutorResult<vllm_cuda::model::deepseek_v2::DeepSeekV2Config> {
    use vllm_cuda::model::deepseek_v2::{DeepSeekV2Config, YarnRopeScaling};

    let hidden_size = hf
        .hidden_size
        .ok_or_else(|| ExecutorError::WorkerInit("missing hidden_size".into()))?;
    let num_attention_heads = hf
        .num_attention_heads
        .ok_or_else(|| ExecutorError::WorkerInit("missing num_attention_heads".into()))?;

    let qk_nope_head_dim = hf
        .extra
        .get("qk_nope_head_dim")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ExecutorError::WorkerInit("missing qk_nope_head_dim".into()))?
        as usize;
    let qk_rope_head_dim = hf
        .extra
        .get("qk_rope_head_dim")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ExecutorError::WorkerInit("missing qk_rope_head_dim".into()))?
        as usize;
    let v_head_dim = hf
        .extra
        .get("v_head_dim")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ExecutorError::WorkerInit("missing v_head_dim".into()))?
        as usize;
    let q_lora_rank = hf
        .extra
        .get("q_lora_rank")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let kv_lora_rank = hf
        .extra
        .get("kv_lora_rank")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ExecutorError::WorkerInit("missing kv_lora_rank".into()))?
        as usize;

    let n_routed_experts = hf
        .extra
        .get("n_routed_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let n_shared_experts = hf
        .extra
        .get("n_shared_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let num_experts_per_tok = hf
        .extra
        .get("num_experts_per_tok")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let first_k_dense_replace = hf
        .extra
        .get("first_k_dense_replace")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let moe_intermediate_size = hf
        .extra
        .get("moe_intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let norm_topk_prob = hf
        .extra
        .get("norm_topk_prob")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let routed_scaling_factor = hf
        .extra
        .get("routed_scaling_factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let n_expert_group = hf
        .extra
        .get("n_group")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let topk_group = hf
        .extra
        .get("topk_group")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let scoring_func = hf
        .extra
        .get("scoring_func")
        .and_then(|v| v.as_str())
        .unwrap_or("softmax")
        .to_string();

    // YaRN rope_scaling (optional)
    let yarn_rope_scaling = hf.extra.get("rope_scaling").and_then(|rs| {
        let rope_type = rs
            .get("rope_type")
            .or_else(|| rs.get("type"))
            .and_then(|v| v.as_str())?;
        if rope_type != "yarn" {
            return None;
        }
        Some(YarnRopeScaling {
            factor: rs.get("factor")?.as_f64()?,
            mscale: rs.get("mscale").and_then(|v| v.as_f64()).unwrap_or(1.0),
            mscale_all_dim: rs
                .get("mscale_all_dim")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            original_max_position_embeddings: rs
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096) as usize,
            beta_fast: rs.get("beta_fast").and_then(|v| v.as_f64()).unwrap_or(32.0),
            beta_slow: rs.get("beta_slow").and_then(|v| v.as_f64()).unwrap_or(1.0),
        })
    });

    Ok(DeepSeekV2Config {
        hidden_size,
        num_attention_heads,
        num_hidden_layers: hf.num_hidden_layers.unwrap_or(60),
        intermediate_size: hf.intermediate_size.unwrap_or(hidden_size * 4),
        vocab_size: hf.vocab_size.unwrap_or(102400),
        max_position_embeddings: hf.max_position_embeddings.unwrap_or(163840),
        rms_norm_eps: hf.rms_norm_eps.unwrap_or(1e-6) as f32,
        rope_theta: hf.rope_theta.unwrap_or(10000.0),
        tie_word_embeddings: hf.tie_word_embeddings.unwrap_or(false),
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        q_lora_rank,
        kv_lora_rank,
        n_routed_experts,
        n_shared_experts,
        num_experts_per_tok,
        first_k_dense_replace,
        moe_intermediate_size,
        norm_topk_prob,
        routed_scaling_factor,
        n_expert_group,
        topk_group,
        scoring_func,
        yarn_rope_scaling,
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
    /// `[max_batch * max_blocks_per_seq]` i32 — page table.
    block_table: PinnedBuf,
    /// Maximum blocks per sequence (computed from max_model_len / block_size).
    max_blocks_per_seq: usize,
    /// `[max_batch]` u32 — D2H token IDs from graph argmax (double-buffered).
    ///
    /// Two pinned buffers alternate per step so that the GPU can write to one
    /// while the CPU reads from the other (deferred D2H pattern).
    host_token_ids: [PinnedBuf; 2],
    /// Which of the two `host_token_ids` buffers is current (0 or 1).
    token_buf_idx: usize,
    /// `[max_batch * 5]` f32 — packed sampling params (temps, top_ks, top_ps, min_ps, randoms).
    sampling_packed: PinnedBuf,
}

impl HostStaging {
    /// Allocate pinned staging buffers for up to `max_batch` decode requests.
    ///
    /// # Safety
    /// Requires active CUDA context.
    unsafe fn new(max_batch: usize, max_blocks_per_seq: usize) -> anyhow::Result<Self> {
        Ok(Self {
            input_ids: unsafe { PinnedBuf::new(max_batch * 4)? },
            positions: unsafe { PinnedBuf::new(max_batch * 4)? },
            slot_mapping: unsafe { PinnedBuf::new(max_batch * 8)? },
            cu_seqlens_q: unsafe { PinnedBuf::new((max_batch + 1) * 4)? },
            seqused_k: unsafe { PinnedBuf::new(max_batch * 4)? },
            block_table: unsafe { PinnedBuf::new(max_batch * max_blocks_per_seq * 4)? },
            max_blocks_per_seq,
            host_token_ids: [unsafe { PinnedBuf::new(max_batch * 4)? }, unsafe {
                PinnedBuf::new(max_batch * 4)?
            }],
            token_buf_idx: 0,
            sampling_packed: unsafe { PinnedBuf::new(max_batch * 5 * 4)? },
        })
    }

    /// Fill block_table pinned buffer from attention metadata block_ids.
    /// Returns a slice of the pinned buffer with `graph_bs * max_blocks_per_seq` elements.
    unsafe fn fill_block_table<'a>(
        &'a self,
        block_ids: &[Vec<usize>],
        graph_bs: usize,
    ) -> &'a [i32] {
        let mbps = self.max_blocks_per_seq;
        let n = graph_bs * mbps;
        let bt = unsafe { self.block_table.slice_mut::<i32>(n) };
        bt.fill(0);
        for (i, blocks) in block_ids.iter().enumerate() {
            for (j, &bid) in blocks.iter().enumerate() {
                if j < mbps {
                    bt[i * mbps + j] = bid as i32;
                }
            }
        }
        &bt[..n]
    }
}

// ---------------------------------------------------------------------------
// CudaWorker
// ---------------------------------------------------------------------------

/// Deferred commit from a previous decode step.
///
/// Stored when the greedy graph fast path defers D2H sync. The token IDs
/// live in one of the double-buffered pinned host staging buffers. The
/// commit is resolved at the start of the next `execute_model_inner` call.
struct PendingCommit {
    /// Which host_token_ids buffer index holds the deferred token IDs.
    buf_idx: usize,
    /// Number of real requests (not graph padding).
    num_reqs: usize,
    /// Per-request IDs (same order as the host buffer).
    req_ids: Vec<String>,
    /// Per-request token count before this step (for `commit_step`).
    token_counts: Vec<usize>,
    /// Per-request flag: true if request had speculative tokens.
    has_spec_tokens: Vec<bool>,
}

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
    /// CUDA graph runner for decode batches (monolithic mode).
    graph_runner: Option<CudaGraphRunner>,
    /// Piecewise CUDA graph runner (attention excluded from graphs).
    piecewise_graph_runner: Option<PiecewiseGraphRunner>,
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

    /// True when the model uses GGML quantized layers (disables CUDA graphs).
    uses_ggml: bool,

    /// Pre-allocated pinned host staging buffers for graph replay.
    /// Initialized after graph capture in `compile_or_warm_up_model`.
    host_staging: Option<HostStaging>,

    // Per-request state.
    token_buffers: HashMap<String, Vec<u32>>,
    /// Per-request prompt length (for discard_request_mask on intermediate prefill chunks).
    prompt_lengths: HashMap<String, usize>,
    /// Per-request block annotations for span-aware RoPE.
    annotation_buffers: HashMap<String, vllm_common::BlockAnnotations>,
    /// Per-request multimodal data (images), only populated for requests
    /// scheduled at first as image-bearing. The vision encoder runs once
    /// per request at first prefill, so the cached entry is removed after
    /// that step (image tokens beyond the first prefill chunk continue to
    /// reference the spliced embeds via the KV cache's stored attention
    /// keys/values — no re-encode needed).
    mm_data_buffers: HashMap<String, vllm_common::MultimodalData>,
    sampling_params_map: HashMap<String, SamplingParams>,
    input_batch: InputBatch,
    preloaded_tokenizer: Option<tokenizers::Tokenizer>,
    /// Set once per thread to avoid redundant `ctx_set_current` driver calls.
    ctx_set_on_thread: bool,
    /// Resolved pooling strategy for embedding mode.
    pooling_strategy: vllm_model::embedding::PoolingStrategy,
    /// Whether executing in pooling mode (--runner pooling).
    is_pooling: bool,

    /// Deferred D2H commit from the previous greedy graph step. Resolved at
    /// the start of the next `execute_model_inner` call.
    pending_commit: Option<PendingCommit>,

    /// Per-request grammar guide state for constrained decoding.
    #[cfg(feature = "guided-decoding")]
    grammar_states: HashMap<String, vllm_model::grammar::GrammarGuide>,
    /// Parser factory for grammar-guided decoding (built once from tokenizer).
    #[cfg(feature = "guided-decoding")]
    grammar_factory: Option<std::sync::Arc<vllm_model::grammar::LlgParserFactory>>,

    /// LogitsProcessor pipeline: persistent GPU state, rebuilt only on batch changes.
    logits_pipeline: Option<LogitsProcessorPipeline>,
    /// Grammar mask processor (separate from pipeline — needs backup logits).
    grammar_processor: GrammarMaskProcessor,
    /// Allowed token IDs processor (separate — needs backup logits like grammar).
    allowed_token_ids_processor: AllowedTokenIdsProcessor,
    /// 🦭 Seal-pad processor: forces pad tokens after EOS for sealed requests.
    seal_pad_processor: SealPadProcessor,
    /// True if batch composition changed this step (triggers BatchUpdate).
    batch_changed: bool,
    /// Ordered request IDs in the current batch (for pipeline update_state).
    batch_req_ids: Vec<String>,
    /// Per-request seeded RNGs for deterministic sampling.
    seeded_rngs: HashMap<String, rand::rngs::StdRng>,

    /// GDN recurrent state pool for Qwen3Next (None for other architectures).
    gdn_state_pool: Option<vllm_cuda::model::qwen3_next::GdnStatePool>,
    /// Qwen3Next config (cached for GDN state pool allocation).
    qwen3_next_config: Option<vllm_cuda::model::qwen3_next::Qwen3NextConfig>,

    /// True when KV cache uses FP8 E4M3 quantization.
    kv_cache_is_fp8: bool,
    /// Compute KV scales dynamically (one-shot: set false after first forward).
    _calculate_kv_scales: bool,
    /// K scale constant (from env or default 1.0).
    _k_scale_constant: f32,
    /// V scale constant (from env or default 1.0).
    _v_scale_constant: f32,
    /// GPU weight allocations tracked for sleep/wake lifecycle.
    /// RAII: `RawGpuMem` calls `driver::mem_free` on drop.
    weight_gpu_allocs: Vec<vllm_cuda::RawGpuMem>,
    /// Saved num_gpu_blocks for re-init after wake.
    num_gpu_blocks_saved: usize,

    // Pipeline parallelism state.
    /// PP NCCL communicator for P2P send/recv between stages.
    #[cfg(feature = "nccl")]
    pp_group: Option<std::sync::Arc<vllm_cuda::nccl::NcclGroup>>,
    /// PP config for this worker (None if PP=1).
    pp_config: Option<vllm_cuda::PpConfig>,
    /// Persistent recv buffer for hidden_states `[max_num_tokens, hidden_size]`.
    /// Pre-allocated on non-first stages for CUDA graph compatibility.
    /// RAII: `RawGpuAlloc` calls `driver::mem_free` on drop.
    pp_recv_hs_buf: Option<vllm_cuda::RawGpuAlloc>,
    /// Persistent recv buffer for residual `[max_num_tokens, hidden_size]`.
    /// RAII: `RawGpuAlloc` calls `driver::mem_free` on drop.
    pp_recv_res_buf: Option<vllm_cuda::RawGpuAlloc>,
    /// Whether a PP send from the previous iteration is pending.
    /// Sync at start of next execute_model to ensure send completed.
    #[allow(dead_code)]
    pp_send_pending: bool,

    /// Optional progress callback for startup initialization.
    /// Used to report layer-by-layer loading progress to the UI.
    #[allow(clippy::type_complexity)]
    progress_callback: Option<std::sync::Arc<dyn Fn(&str) + Send + Sync>>,
}

// Safety: CudaWorker contains raw GPU pointers (via GpuDevice, model weights,
// KV cache, PP buffers, CUDA graphs) and raw pointer fields in OwnedTensor /
// RawGpuAlloc / RawGpuMem. All GPU resources are allocated on a single CUDA
// context and accessed exclusively from the worker thread. Send is required
// because the worker is created on the main thread and moved to its dedicated
// worker thread via the executor's spawn.
unsafe impl Send for CudaWorker {}

impl CudaWorker {
    pub fn new(config: CudaWorkerConfig) -> Self {
        let is_pooling = config.is_pooling;
        let kv_cache_is_fp8 = config.kv_cache_dtype == "fp8_e4m3" || config.kv_cache_dtype == "fp8";
        let calculate_kv_scales = config.calculate_kv_scales;
        // Scale constants: match Python's defaults. Override via env vars.
        let k_scale_constant = std::env::var("VLLM_FP8_K_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0_f32);
        let v_scale_constant = std::env::var("VLLM_FP8_V_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0_f32);
        let seal_pad_processor = SealPadProcessor::new(
            config.eos_token_ids.clone(),
            0, // pad_token: default 0, updated during model load from tokenizer
            config.block_size,
        );
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
            piecewise_graph_runner: None,
            prefill_graph_runner: None,
            last_graph_batch_size: None,
            graph_metadata_valid: false,
            uses_ggml: false,
            host_staging: None,
            token_buffers: HashMap::new(),
            prompt_lengths: HashMap::new(),
            annotation_buffers: HashMap::new(),
            mm_data_buffers: HashMap::new(),
            sampling_params_map: HashMap::new(),
            input_batch: InputBatch::new(),
            preloaded_tokenizer: None,
            ctx_set_on_thread: false,
            pooling_strategy: vllm_model::embedding::PoolingStrategy::Last,
            is_pooling,
            pending_commit: None,
            #[cfg(feature = "guided-decoding")]
            grammar_states: HashMap::new(),
            #[cfg(feature = "guided-decoding")]
            grammar_factory: None,
            logits_pipeline: None,
            grammar_processor: GrammarMaskProcessor::new(),
            allowed_token_ids_processor: AllowedTokenIdsProcessor::new(),
            seal_pad_processor,
            batch_changed: false,
            batch_req_ids: Vec::new(),
            seeded_rngs: HashMap::new(),
            gdn_state_pool: None,
            qwen3_next_config: None,
            kv_cache_is_fp8,
            _calculate_kv_scales: calculate_kv_scales,
            _k_scale_constant: k_scale_constant,
            _v_scale_constant: v_scale_constant,
            weight_gpu_allocs: Vec::new(),
            num_gpu_blocks_saved: 0,
            #[cfg(feature = "nccl")]
            pp_group: None,
            pp_config: None,
            pp_recv_hs_buf: None,
            pp_recv_res_buf: None,
            pp_send_pending: false,
            progress_callback: None,
        }
    }

    /// Set progress callback for reporting model loading progress.
    pub fn set_progress_callback<F>(&mut self, callback: std::sync::Arc<F>)
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.progress_callback = Some(callback);
    }

    /// Expose the GpuDevice (for NCCL stream access during TP init).
    pub fn device_ref(&self) -> Option<&GpuDevice> {
        self.device.as_ref()
    }

    /// Inject NCCL process group into the loaded model for TP communication.
    #[cfg(feature = "nccl")]
    pub fn set_tp_group(&mut self, group: std::sync::Arc<vllm_cuda::nccl::NcclGroup>) {
        if let Some(ref mut model) = self.model {
            model.set_tp_group(group);
        }
    }

    /// Inject NCCL process group for PP P2P communication.
    #[cfg(feature = "nccl")]
    pub fn set_pp_group(&mut self, group: std::sync::Arc<vllm_cuda::nccl::NcclGroup>) {
        self.pp_group = Some(group);
    }

    /// Set PP config for this worker.
    pub fn set_pp_config(&mut self, pp: vllm_cuda::PpConfig) {
        self.pp_config = Some(pp);
    }

    /// Allocate persistent PP recv buffers after model is loaded and
    /// hidden_size is known. Called during init on non-first PP stages.
    pub fn allocate_pp_recv_buffers(&mut self) {
        let pp = match self.pp_config {
            Some(pp) if !pp.is_first_stage() => pp,
            _ => return, // No recv buffers needed on first stage or no PP.
        };
        let _ = pp; // used just for gating

        let model = self.model.as_ref().expect("model must be loaded first");
        let hidden_size = model.hidden_size();
        let max_tokens = self.config.max_num_batched_tokens;
        let gpu_dtype = self.model_dtype;

        unsafe {
            self.pp_recv_hs_buf = Some(
                vllm_cuda::RawGpuAlloc::new(&[max_tokens, hidden_size], gpu_dtype)
                    .expect("failed to allocate PP recv hs buffer"),
            );

            self.pp_recv_res_buf = Some(
                vllm_cuda::RawGpuAlloc::new(&[max_tokens, hidden_size], gpu_dtype)
                    .expect("failed to allocate PP recv res buffer"),
            );
        }
    }

    /// Recv intermediate tensors from previous PP stage into persistent buffers,
    /// then copy the active slice `[:num_tokens]` into OwnedTensors for the model.
    ///
    /// Returns `(hidden_states, residual)` as OwnedTensors.
    /// Takes explicit field references to avoid &mut self borrow conflicts.
    #[cfg(feature = "nccl")]
    fn pp_recv_intermediates(
        pp_group: &vllm_cuda::nccl::NcclGroup,
        pp_recv_hs_buf: &vllm_cuda::RawGpuAlloc,
        pp_recv_res_buf: &vllm_cuda::RawGpuAlloc,
        pp: vllm_cuda::PpConfig,
        num_tokens: usize,
        device: &mut GpuDevice,
    ) -> ExecutorResult<(vllm_cuda::OwnedTensor, vllm_cuda::OwnedTensor)> {
        let hs_buf = pp_recv_hs_buf;
        let res_buf = pp_recv_res_buf;

        let hidden_size = hs_buf.shape()[1] as usize;
        let dtype = hs_buf.dtype();
        let prev_pp_rank = pp.pp_rank - 1;

        // Slice persistent buffers to [:num_tokens].
        let hs_slice = hs_buf.narrow_dim0(0, num_tokens);
        let res_slice = res_buf.narrow_dim0(0, num_tokens);

        // Non-blocking recv into persistent buffers (on PP NCCL stream).
        unsafe {
            pp_group
                .recv(hs_slice, prev_pp_rank)
                .map_err(|e| ExecutorError::WorkerExecution(format!("PP recv hs: {e}")))?;
            pp_group
                .recv(res_slice, prev_pp_rank)
                .map_err(|e| ExecutorError::WorkerExecution(format!("PP recv res: {e}")))?;
        }

        // Sync the NCCL stream to ensure recv completes before forward.
        unsafe {
            vllm_cuda::driver::stream_synchronize(pp_group.stream())
                .map_err(|e| ExecutorError::WorkerExecution(format!("PP recv stream sync: {e}")))?;
        }

        // Copy received data into OwnedTensors from caching allocator.
        let hs_owned = device
            .caching
            .alloc_tensor(&[num_tokens, hidden_size], dtype);
        let res_owned = device
            .caching
            .alloc_tensor(&[num_tokens, hidden_size], dtype);

        let nbytes = num_tokens * hidden_size * dtype.size_bytes();
        unsafe {
            vllm_cuda::driver::memcpy_dtod_async(
                hs_owned.as_gpu_tensor().raw_ptr(),
                hs_slice.raw_ptr(),
                nbytes,
                device.compute_stream,
            )
            .map_err(|e| ExecutorError::WorkerExecution(format!("PP copy hs: {e}")))?;
            vllm_cuda::driver::memcpy_dtod_async(
                res_owned.as_gpu_tensor().raw_ptr(),
                res_slice.raw_ptr(),
                nbytes,
                device.compute_stream,
            )
            .map_err(|e| ExecutorError::WorkerExecution(format!("PP copy res: {e}")))?;
        }

        Ok((hs_owned, res_owned))
    }

    /// Execute forward pass for a non-last PP stage:
    /// 1. Sync previous sends
    /// 2. Recv intermediates (if not first stage)
    /// 3. Run forward_pp (eager)
    /// 4. Send intermediates to next stage
    /// 5. Commit step + return dummy output
    #[cfg(feature = "nccl")]
    fn execute_pp_non_last_stage(
        &mut self,
        pp: vllm_cuda::PpConfig,
        prepared: PreparedInputs,
        block_size: usize,
    ) -> ExecutorResult<ModelRunnerOutput> {
        let num_reqs = prepared.req_inputs.len();
        let total_tokens = prepared.flat_token_ids.len();
        let next_pp_rank = pp.pp_rank + 1;
        // Step 0: Sync previous PP send (if pending).
        if self.pp_send_pending {
            let pp_group = self.pp_group.as_ref().unwrap();
            unsafe {
                vllm_cuda::driver::stream_synchronize(pp_group.stream())
                    .map_err(|e| ExecutorError::WorkerExecution(format!("PP send sync: {e}")))?;
            }
            self.pp_send_pending = false;
        }

        // Step 1: Recv intermediates (if not first stage).
        // Done before destructuring to avoid borrow conflicts.
        let intermediate = if !pp.is_first_stage() {
            let pp_group = self.pp_group.as_ref().unwrap();
            let hs_buf = self.pp_recv_hs_buf.as_ref().unwrap();
            let res_buf = self.pp_recv_res_buf.as_ref().unwrap();
            let device = self
                .device
                .as_mut()
                .ok_or_else(|| ExecutorError::WorkerExecution("device not initialized".into()))?;
            Some(Self::pp_recv_intermediates(
                pp_group,
                hs_buf,
                res_buf,
                pp,
                total_tokens,
                device,
            )?)
        } else {
            None
        };

        let (model, kv_cache, device) = match (&self.model, &self.kv_cache, &mut self.device) {
            (Some(m), Some(kv), Some(d)) => (m, kv, d),
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "model, KV cache, or device not initialized".into(),
                ));
            }
        };

        // Step 2: Prepare GPU inputs.
        let gpu_input_ids = Self::h2d_u32(&prepared.flat_token_ids, device)?;
        let gpu_positions = Self::h2d_u32(&prepared.flat_positions, device)?;
        let (slot_mapping, cu_seqlens_q, seqused_k, block_table, max_seqlen_q, max_seqlen_k) =
            Self::build_attention_tensors(&prepared.attn_meta, block_size, device)?;

        // For prefills, compute last_token_indices.
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

        let input_ids = if pp.is_first_stage() {
            Some(gpu_input_ids.view())
        } else {
            None
        };

        // Step 3: Forward pass.
        let result = unsafe {
            model.forward_pp(
                input_ids,
                intermediate,
                gpu_positions.view(),
                slot_mapping.view(),
                cu_seqlens_q.view(),
                seqused_k.view(),
                block_table.view(),
                max_seqlen_q,
                max_seqlen_k,
                kv_cache,
                device,
                last_token_indices.as_ref().map(|t| t.view()),
            )
        };

        // Step 4: Send intermediates to next stage.
        let (hs, res) = match result {
            vllm_cuda::model::llama::ForwardOutput::Intermediate {
                hidden_states,
                residual,
            } => (hidden_states, residual),
            vllm_cuda::model::llama::ForwardOutput::Logits(_) => {
                unreachable!("non-last PP stage should return Intermediate");
            }
        };

        let pp_group = self.pp_group.as_ref().unwrap();
        // Sync compute stream before sending on NCCL stream — ensures forward
        // output is complete before NCCL reads from the tensors.
        unsafe {
            vllm_cuda::driver::stream_synchronize(device.compute_stream).map_err(|e| {
                ExecutorError::WorkerExecution(format!("PP compute sync before send: {e}"))
            })?;
        }
        unsafe {
            pp_group
                .send(hs.as_gpu_tensor(), next_pp_rank)
                .map_err(|e| ExecutorError::WorkerExecution(format!("PP send hs: {e}")))?;
            pp_group
                .send(res.as_gpu_tensor(), next_pp_rank)
                .map_err(|e| ExecutorError::WorkerExecution(format!("PP send res: {e}")))?;
        }
        self.pp_send_pending = true;
        // Drop OwnedTensors after sends are enqueued (non-blocking).
        // The NCCL stream will hold a reference to the GPU memory.
        drop(hs);
        drop(res);

        // Step 5: Commit step for all requests (dummy token 0 — last stage
        // broadcasts real tokens back in async scheduling mode; for sync
        // scheduling the scheduler provides them via CachedRequestData).
        for req_slice in &prepared.req_inputs {
            self.input_batch.commit_step(
                &req_slice.req_id,
                &[0], // dummy token
                req_slice.token_count,
                false,
            );
        }
        self.input_batch.reclaim_buffers(prepared);

        // Return dummy output — only the output_rank's result matters.
        Ok(ModelRunnerOutput::from_token_map(HashMap::new()))
    }

    /// Update max_num_batched_tokens (used for GPU-aware auto-detection).
    pub fn set_max_num_batched_tokens(&mut self, value: usize) {
        self.config.max_num_batched_tokens = value;
    }

    /// Expose the HF config after load_model.
    pub fn hf_config(&self) -> Option<&HfModelConfig> {
        self.hf_config.as_ref()
    }

    /// Maximum blocks per sequence: cdiv(max_model_len, block_size).
    /// Matches Python's `BlockTables.__init__` computation.
    fn max_blocks_per_seq(&self) -> usize {
        let max_model_len = self
            .hf_config
            .as_ref()
            .and_then(|c| c.max_position_embeddings)
            .unwrap_or(131072);
        max_model_len.div_ceil(self.config.block_size)
    }

    /// Expose the model directory after load_model.
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }

    /// Bytes per element for the model's KV cache dtype.
    pub fn resolved_dtype_elem_bytes(&self) -> usize {
        self.model_dtype.size_bytes()
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
            info!("CudaWorker: no tokenizer.json found, grammar-guided decoding unavailable");
            return;
        }
        let tokenizer_bytes = match std::fs::read(&tokenizer_path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    "CudaWorker: failed to read tokenizer.json for grammar factory: {e}"
                );
                return;
            }
        };

        match vllm_model::grammar::build_parser_factory(&tokenizer_bytes) {
            Ok(factory) => {
                info!("CudaWorker: grammar parser factory built");
                self.grammar_factory = Some(factory);
            }
            Err(e) => {
                tracing::warn!("CudaWorker: failed to build grammar parser factory: {e}");
            }
        }
    }

    /// Resolve model path: local dir, local GGUF file, or HF download.
    ///
    /// Returns a `PathBuf` that is either:
    /// - A directory containing safetensors + config.json (normal path)
    /// - A `.gguf` file path (GGUF path — load_model detects this)
    fn resolve_model_path(&self) -> ExecutorResult<PathBuf> {
        let path = Path::new(&self.config.model_path);

        // Local .gguf file.
        if path.is_file() && path.extension().is_some_and(|e| e == "gguf") {
            return Ok(path.to_path_buf());
        }

        // Local directory.
        if path.is_dir() {
            return Ok(path.to_path_buf());
        }

        info!(
            "Downloading model from HuggingFace Hub: {}",
            self.config.model_path
        );
        let mut builder = hf_hub::api::sync::ApiBuilder::from_env();
        if let Some(ref token) = self.config.hf_token {
            builder = builder.with_token(Some(token.clone()));
        }
        let api = builder
            .build()
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to build HF API: {e}")))?;
        let repo = api.model(self.config.model_path.clone());

        // GGUF download: explicit filename or auto-detect from repo.
        let gguf_filename = self.config.gguf_file.clone().or_else(|| {
            // Auto-detect: if model name looks like a GGUF repo, find smallest Q4_K_M file.
            if !self.config.model_path.to_ascii_uppercase().contains("GGUF") {
                return None;
            }
            let info = repo.info().ok()?;
            let mut gguf_files: Vec<_> = info
                .siblings
                .iter()
                .filter(|s| s.rfilename.ends_with(".gguf"))
                .collect();
            if gguf_files.is_empty() {
                return None;
            }
            // Prefer Q4_K_M, then Q4_K_S, then any Q4, then smallest file.
            for pattern in &["Q4_K_M", "Q4_K_S", "Q4_K", "Q4_0", "Q8_0"] {
                if let Some(f) = gguf_files.iter().find(|s| s.rfilename.contains(pattern)) {
                    return Some(f.rfilename.clone());
                }
            }
            // Fallback: first GGUF file alphabetically.
            gguf_files.sort_by(|a, b| a.rfilename.cmp(&b.rfilename));
            Some(gguf_files[0].rfilename.clone())
        });
        if let Some(ref gguf_file) = gguf_filename {
            info!("Downloading GGUF file: {gguf_file}");
            let gguf_path = repo.get(gguf_file).map_err(|e| {
                ExecutorError::WorkerInit(format!("failed to download GGUF {gguf_file}: {e}"))
            })?;
            // Best-effort tokenizer download.
            let _ = repo.get("tokenizer.json");
            let _ = repo.get("tokenizer_config.json");
            return Ok(gguf_path);
        }

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

        Err(ExecutorError::WorkerInit(format!(
            "no safetensors weights found for {}",
            self.config.model_path
        )))
    }

    /// Resolve a LoRA adapter path: local directory or HF Hub download.
    fn resolve_adapter_path(&self, adapter_path: &str) -> ExecutorResult<PathBuf> {
        let path = Path::new(adapter_path);
        if path.is_dir() {
            return Ok(path.to_path_buf());
        }

        // Download from HuggingFace Hub.
        info!("Downloading LoRA adapter from HuggingFace Hub: {adapter_path}");
        let mut builder = hf_hub::api::sync::ApiBuilder::from_env();
        if let Some(ref token) = self.config.hf_token {
            builder = builder.with_token(Some(token.clone()));
        }
        let api = builder
            .build()
            .map_err(|e| ExecutorError::WorkerInit(format!("HF API: {e}")))?;
        let repo = api.model(adapter_path.to_string());

        let config_path = repo.get("adapter_config.json").map_err(|e| {
            ExecutorError::WorkerInit(format!("failed to download adapter_config.json: {e}"))
        })?;
        let adapter_dir = config_path.parent().unwrap().to_path_buf();

        repo.get("adapter_model.safetensors").map_err(|e| {
            ExecutorError::WorkerInit(format!("failed to download adapter_model.safetensors: {e}"))
        })?;

        Ok(adapter_dir)
    }

    /// H2D copy a u32 slice into a caching-allocator tensor.
    fn h2d_u32(data: &[u32], device: &mut GpuDevice) -> ExecutorResult<OwnedTensor> {
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
        Ok(t)
    }

    /// H2D copy an i32 slice into a caching-allocator tensor.
    fn h2d_i32(data: &[i32], device: &mut GpuDevice) -> ExecutorResult<OwnedTensor> {
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
        Ok(t)
    }

    fn h2d_i64(data: &[i64], device: &mut GpuDevice) -> ExecutorResult<OwnedTensor> {
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
        Ok(t)
    }

    /// Build attention metadata tensors from `AttentionMetadata`.
    ///
    /// Returns `(slot_mapping, cu_seqlens_q, seqused_k, block_table, max_seqlen_q, max_seqlen_k)`.
    /// `seqused_k` has per-sequence K lengths `[num_reqs]` for the paged FA2 splitkv kernel.
    fn build_attention_tensors(
        meta: &vllm_model::AttentionMetadata,
        block_size: usize,
        device: &mut GpuDevice,
    ) -> ExecutorResult<(
        OwnedTensor,
        OwnedTensor,
        OwnedTensor,
        OwnedTensor,
        usize,
        usize,
    )> {
        let num_reqs = meta.num_reqs;

        // Invariant checks on attention metadata.
        debug_assert_eq!(meta.query_start_loc.len(), num_reqs + 1);
        debug_assert_eq!(meta.seq_lens.len(), num_reqs);
        debug_assert_eq!(meta.q_lens.len(), num_reqs);
        debug_assert_eq!(meta.block_ids.len(), num_reqs);
        debug_assert_eq!(meta.tokens_before.len(), num_reqs);
        debug_assert_eq!(
            meta.q_lens.iter().sum::<usize>(),
            meta.total_tokens,
            "sum(q_lens) != total_tokens"
        );
        debug_assert_eq!(*meta.query_start_loc.last().unwrap(), meta.total_tokens);
        for i in 0..num_reqs {
            debug_assert!(
                meta.seq_lens[i] >= meta.q_lens[i],
                "req {i}: seq_len={} < q_len={}",
                meta.seq_lens[i],
                meta.q_lens[i]
            );
            let needed_blocks = meta.seq_lens[i].div_ceil(block_size);
            debug_assert!(
                meta.block_ids[i].len() >= needed_blocks,
                "req {i}: block_ids.len()={} < needed_blocks={} for seq_len={} block_size={} \
                 tokens_before={} q_len={}",
                meta.block_ids[i].len(),
                needed_blocks,
                meta.seq_lens[i],
                block_size,
                meta.tokens_before[i],
                meta.q_lens[i]
            );
        }

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
            let mut t = Self::h2d_i32(&block_table, device)?;
            unsafe { t.reshape(&[num_reqs, max_blocks], GpuDType::I32) };
            t
        } else {
            device.caching.alloc_tensor(&[0], GpuDType::I32)
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

    /// Run the vision encoder for any active request whose first prefill
    /// chunk includes its image placeholders. Returns
    /// `(mm_embeds, embed_patches)` ready to thread into
    /// `MmForwardInputs` — `mm_embeds` is `[total_mm_tokens, hidden]`,
    /// `embed_patches[i].token_offset` is the destination row in the
    /// flat input-id sequence (post batch concatenation).
    ///
    /// Returns `None` when there's nothing to encode this step:
    /// - non-Ferrite arch, or Ferrite arch with no MM handle (text-only),
    /// - no req in the active batch carries `mm_data`,
    /// - no MM-bearing req is at its first scheduling step
    ///   (`tokens_before == 0`).
    ///
    /// MVP: assumes the prefill chunk for an MM-bearing req covers ALL
    /// its image placeholders (no chunked-prefill split across image
    /// boundaries). For typical Qwen2-VL-2B prompts (256 tokens / image)
    /// this holds inside a single 2048-token prefill chunk; the chunked
    /// case is a follow-up.
    ///
    /// # Safety
    /// `device` must be the live CUDA device the encoder kernels
    /// launch on. The returned `OwnedTensor` is on `device.compute_stream`
    /// — the caller must keep it alive across the language-side
    /// `model.forward` call so the `Instruction::Embed::eval` D2D
    /// splice still has live source memory.
    unsafe fn run_mm_vision_forward(
        model: &CudaModel,
        mm_data_buffers: &HashMap<String, vllm_common::MultimodalData>,
        prepared: &PreparedInputs,
        device: &mut GpuDevice,
    ) -> Option<(OwnedTensor, Vec<ferrite_forward::EmbedPatch>)> {
        let CudaModel::Ferrite(fm) = model else {
            return None;
        };
        let mm = fm.mm.as_ref()?;
        let meta = &prepared.attn_meta;
        let mut pixel_inputs: Vec<ferrite_forward::PixelInput<'_>> = Vec::new();
        let mut placeholders: Vec<ferrite_forward::EmbedPatch> = Vec::new();
        for (i, req) in prepared.req_inputs.iter().enumerate() {
            let Some(mm_data) = mm_data_buffers.get(&req.req_id) else {
                continue;
            };
            // Only run vision_forward for the first prefill chunk —
            // subsequent decode steps consume already-spliced KV cache.
            if meta.tokens_before[i] != 0 {
                continue;
            }
            let batch_offset = meta.query_start_loc[i] as u32;
            for (img, ph) in mm_data.images.iter().zip(mm_data.image_placeholders.iter()) {
                pixel_inputs.push(ferrite_forward::PixelInput {
                    pixels: &img.pixels,
                    height: img.height as u32,
                    width: img.width as u32,
                });
                placeholders.push(ferrite_forward::EmbedPatch {
                    token_offset: batch_offset + ph.offset as u32,
                    length: ph.length as u32,
                    // Filled by `vision_forward` from per-image
                    // `grid_thw` + `spatial_merge_size`. Default zero
                    // here is invalid for MRoPE; the encoder must
                    // populate before the executor reads them.
                    grid_t: 0,
                    grid_h_merged: 0,
                    grid_w_merged: 0,
                });
            }
        }
        if pixel_inputs.is_empty() {
            return None;
        }
        let (out, patches) = unsafe { mm.vision_forward(&pixel_inputs, &placeholders, device) };
        Some((out, patches))
    }

    /// Build per-req seq-space MM info for **every** MM-bearing req in
    /// the active batch — including ones whose vision encoder already
    /// ran on a prior step (cached prefix). Empty inner `Vec` for any
    /// req without mm_data. Returns an all-empty outer `Vec` when no
    /// req carries mm_data (text-only batch → caller falls back to 1D
    /// positions, byte-identical to pre-MM behavior).
    ///
    /// Tuple is `(seq_offset, length, grid_t, grid_h_merged, grid_w_merged)`
    /// in seq space (offset relative to the req's full sequence start,
    /// not batch start). [`Self::build_mrope_positions_2d`] walks the
    /// entire seq — including the cached prefix — so the cursor lands
    /// at the same position the encoder used to write KV the first
    /// time around.
    fn build_per_req_mm_seq_info(
        model: &CudaModel,
        mm_data_buffers: &HashMap<String, vllm_common::MultimodalData>,
        prepared: &PreparedInputs,
    ) -> Vec<Vec<SeqMmInfo>> {
        let mut per_req: Vec<Vec<SeqMmInfo>> = vec![Vec::new(); prepared.req_inputs.len()];
        let CudaModel::Ferrite(fm) = model else {
            return per_req;
        };
        let Some(mm) = fm.mm.as_ref() else {
            return per_req;
        };
        let mut any = false;
        for (i, req) in prepared.req_inputs.iter().enumerate() {
            let Some(mm_data) = mm_data_buffers.get(&req.req_id) else {
                continue;
            };
            if mm_data.images.is_empty() {
                continue;
            }
            let pixel_inputs: Vec<ferrite_forward::PixelInput<'_>> = mm_data
                .images
                .iter()
                .map(|img| ferrite_forward::PixelInput {
                    pixels: &img.pixels,
                    height: img.height as u32,
                    width: img.width as u32,
                })
                .collect();
            let grids = mm.embed_patch_grids(&pixel_inputs);
            for (ph, &(gt, mh, mw)) in mm_data.image_placeholders.iter().zip(grids.iter()) {
                per_req[i].push((ph.offset as u32, ph.length as u32, gt, mh, mw));
            }
            // Sort by seq offset so the walk visits patches in order.
            per_req[i].sort_by_key(|t| t.0);
            any = true;
        }
        if !any {
            return Vec::new();
        }
        per_req
    }

    /// Build `[3, n_tokens]` u32 MRoPE positions for an image-bearing
    /// batch. Mirrors Python vLLM's
    /// `Qwen2VLForConditionalGeneration.get_input_positions_tensor`:
    /// text tokens get `(cursor, cursor, cursor)` with `cursor`
    /// incrementing by 1; image tokens scan the post-spatial-merge
    /// grid in `(t, h, w)` row-major and emit `(st+t, st+h, st+w)`
    /// where `st` is the cursor at image entry; cursor advances by
    /// `max(grid_t, grid_h_merged, grid_w_merged)` after each image.
    ///
    /// Walks **every** seq position from 0 to `tokens_before + q_len`,
    /// advancing the cursor through preceding image patches even when
    /// they lie entirely in the cached prefix. Positions are written
    /// only for tokens in the current q range — but the cursor that
    /// gets written is the same one the encoder used to encode the
    /// cached KV, so `cached==prompt-1` (Bug 3) decodes correctly.
    /// Reqs with no patches get the linear cursor for every row —
    /// numerically identical to text-only 1D rope through the kernel's
    /// broadcast path, but in the 2D shape so a mixed batch (text req +
    /// image req) can share one positions tensor.
    fn build_mrope_positions_2d(
        prepared: &PreparedInputs,
        per_req_mm: &[Vec<SeqMmInfo>],
    ) -> Vec<u32> {
        let n_tokens = prepared.flat_positions.len();
        let mut t_row = vec![0u32; n_tokens];
        let mut h_row = vec![0u32; n_tokens];
        let mut w_row = vec![0u32; n_tokens];
        let meta = &prepared.attn_meta;

        for (req_idx, req_patches) in per_req_mm.iter().enumerate() {
            let req_start_batch = meta.query_start_loc[req_idx];
            let q_len = meta.q_lens[req_idx] as u32;
            let num_computed = meta.tokens_before[req_idx] as u32;
            let q_seq_end = num_computed + q_len;

            let mut cursor = 0u32;
            let mut seq_pos = 0u32;
            let mut patch_iter = req_patches.iter().peekable();

            while seq_pos < q_seq_end {
                if let Some(&&(s_off, length, gt, mh, mw)) = patch_iter.peek()
                    && s_off == seq_pos
                {
                    let st = cursor;
                    let stride_hw = mh * mw;
                    for img_idx in 0..length {
                        let pos_seq = seq_pos + img_idx;
                        if pos_seq >= num_computed && pos_seq < q_seq_end {
                            let local_q = (pos_seq - num_computed) as usize;
                            let flat_idx = req_start_batch + local_q;
                            let t = img_idx / stride_hw;
                            let rem = img_idx % stride_hw;
                            let h = rem / mw;
                            let w = rem % mw;
                            t_row[flat_idx] = st + t;
                            h_row[flat_idx] = st + h;
                            w_row[flat_idx] = st + w;
                        }
                    }
                    cursor = st + gt.max(mh).max(mw);
                    seq_pos += length;
                    patch_iter.next();
                    continue;
                }
                if seq_pos >= num_computed {
                    let local_q = (seq_pos - num_computed) as usize;
                    let flat_idx = req_start_batch + local_q;
                    t_row[flat_idx] = cursor;
                    h_row[flat_idx] = cursor;
                    w_row[flat_idx] = cursor;
                }
                cursor += 1;
                seq_pos += 1;
            }
        }

        let mut out = Vec::with_capacity(3 * n_tokens);
        out.extend_from_slice(&t_row);
        out.extend_from_slice(&h_row);
        out.extend_from_slice(&w_row);
        out
    }

    /// Build GDN state_indices and cu_seqlens for Qwen3Next forward.
    ///
    /// Returns (gdn_state_indices, gdn_cu_seqlens, num_seqs) on GPU.
    /// state_indices: [num_seqs] i32 — slot index per sequence (= batch index).
    /// cu_seqlens: [num_seqs + 1] i32 — cumulative query lengths.
    fn build_gdn_tensors(
        meta: &vllm_model::AttentionMetadata,
        device: &mut GpuDevice,
    ) -> ExecutorResult<(OwnedTensor, OwnedTensor, usize)> {
        let num_seqs = meta.num_reqs;

        // State indices: sequence i uses slot i in the GDN state pool.
        let state_indices: Vec<i32> = (0..num_seqs as i32).collect();
        let gpu_state_indices = Self::h2d_i32(&state_indices, device)?;

        // cu_seqlens for GDN: same as attention cu_seqlens_q.
        let cu_seqlens: Vec<i32> = meta.query_start_loc.iter().map(|&x| x as i32).collect();
        let gpu_cu_seqlens = Self::h2d_i32(&cu_seqlens, device)?;

        Ok((gpu_state_indices, gpu_cu_seqlens, num_seqs))
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
        strategy: vllm_model::embedding::PoolingStrategy,
        device: &mut GpuDevice,
    ) -> ExecutorResult<EmbeddingData> {
        use vllm_model::embedding::PoolingStrategy;

        let hidden_size = hidden_states.dim(1);

        // AllTokens: D2H all rows, L2-normalize each, return as Multi.
        if strategy == PoolingStrategy::AllTokens {
            let all_f32 = Self::logits_to_cpu(hidden_states, device)?;
            let mut rows = Vec::with_capacity(num_tokens);
            for row in 0..num_tokens {
                let start = row * hidden_size;
                let end = start + hidden_size;
                let mut vec: Vec<f32> = all_f32[start..end].to_vec();
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut vec {
                        *v /= norm;
                    }
                }
                rows.push(vec);
            }
            return Ok(EmbeddingData::Multi(rows));
        }

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
                    return Ok(EmbeddingData::Single(mean));
                }
            }
            PoolingStrategy::AllTokens => unreachable!("handled by early return"),
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
            Ok(EmbeddingData::Single(
                f32_vec.into_iter().map(|x| x / norm).collect(),
            ))
        } else {
            Ok(EmbeddingData::Single(f32_vec))
        }
    }

    /// D2H copy token IDs using pinned staging if available.
    /// Enqueue async D2H + sync, returning token IDs. Uses the pinned double-
    /// buffer at `buf_idx` when staging is available.
    fn d2h_token_ids_sync(
        staging: Option<&HostStaging>,
        buf_idx: usize,
        gpu_tensor: &GpuTensor,
        num_reqs: usize,
        device: &mut GpuDevice,
    ) -> ExecutorResult<Vec<u32>> {
        if let Some(stg) = staging {
            // Async D2H: compute→event→transfer_stream→D2H→event→sync event.
            unsafe {
                device.async_d2h(
                    stg.host_token_ids[buf_idx].ptr(),
                    gpu_tensor.raw_ptr() as *const u8,
                    num_reqs * 4,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("async D2H token ids: {e}")))?;
            device
                .sync_d2h()
                .map_err(|e| ExecutorError::WorkerExecution(format!("sync d2h: {e}")))?;
            Ok(unsafe { stg.host_token_ids[buf_idx].slice::<u32>(num_reqs) }.to_vec())
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

    /// Enqueue async D2H without sync. Returns the buffer index used.
    /// Caller must sync via `device.sync_d2h()` before reading the buffer.
    fn d2h_token_ids_async(
        staging: &HostStaging,
        buf_idx: usize,
        gpu_tensor: &GpuTensor,
        num_reqs: usize,
        device: &GpuDevice,
    ) -> ExecutorResult<()> {
        unsafe {
            device.async_d2h(
                staging.host_token_ids[buf_idx].ptr(),
                gpu_tensor.raw_ptr() as *const u8,
                num_reqs * 4,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("async D2H token ids: {e}")))?;
        Ok(())
    }
    /// Full GPU sampling pipeline: grammar mask → logit processors → sample → logprobs.
    /// No CPU fallback — everything stays on GPU, matching Python vLLM exactly.
    ///
    /// Uses the `LogitsProcessorPipeline` for persistent GPU state (logit_bias,
    /// penalties, min_tokens) and a separate `GrammarMaskProcessor` for grammar.
    #[allow(clippy::too_many_arguments)]
    fn gpu_sample_and_finalize(
        sampling_params_map: &HashMap<String, SamplingParams>,
        #[cfg(feature = "guided-decoding")] grammar_states: &mut HashMap<
            String,
            vllm_model::grammar::GrammarGuide,
        >,
        grammar_processor: &GrammarMaskProcessor,
        allowed_token_ids_processor: &AllowedTokenIdsProcessor,
        seal_pad_processor: &SealPadProcessor,
        logits_pipeline: Option<&LogitsProcessorPipeline>,
        seeded_rngs: &mut HashMap<String, rand::rngs::StdRng>,
        host_staging: &Option<HostStaging>,
        input_batch: &mut InputBatch,
        token_buffers: &mut HashMap<String, Vec<u32>>,
        prompt_lengths: &HashMap<String, usize>,
        logits: GpuTensor,
        prepared: PreparedInputs,
        device: &mut GpuDevice,
        all_greedy: bool,
        vocab_size: usize,
    ) -> ExecutorResult<ModelRunnerOutput> {
        use rand::Rng;
        let num_reqs = prepared.req_inputs.len();
        let total_tokens = prepared.flat_token_ids.len();

        let any_spec_decode = prepared
            .req_inputs
            .iter()
            .any(|r| !r.spec_token_ids.is_empty());

        // --- Spec decode path: greedy rejection sampling ---
        // When any request has draft tokens, logits shape is [total_tokens, vocab_size]
        // (all positions). We argmax all rows, then do CPU greedy rejection.
        if any_spec_decode && all_greedy {
            return Self::spec_decode_greedy_sample(
                logits,
                total_tokens,
                prepared,
                device,
                host_staging,
                input_batch,
                token_buffers,
            );
        }

        // Helper: get a per-request random seed (u32) for GPU sampling kernels.
        //
        // For requests with an explicit seed: use the per-request seeded RNG.
        // For unseeded requests: derive from hash(req_id, position) — this is
        // stateless and identical on all TP ranks, unlike a stateful RNG which
        // can desync if ranks call gen_seed different numbers of times during
        // prefill chunking.
        //
        // The GPU Gumbel kernel uses this as a Philox RNG seed to generate
        // per-element randomness (matching Python vLLM's tl.rand approach).
        let positions = &prepared.flat_positions;
        let mut req_offset = 0usize;
        let mut gen_seed = |req_id: &str| -> u32 {
            if let Some(rng) = seeded_rngs.get_mut(req_id) {
                rng.r#gen::<u32>()
            } else {
                // Stateless hash: combine req_id bytes with position (step
                // counter). Must be deterministic across TP ranks — we use a
                // simple FNV-1a hash with a fixed seed (NOT DefaultHasher which
                // uses a random per-process seed for DOS protection).
                let pos = positions.get(req_offset).copied().unwrap_or(0);
                req_offset += 1;
                let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
                for b in req_id.as_bytes() {
                    h ^= *b as u64;
                    h = h.wrapping_mul(0x100000001b3); // FNV prime
                }
                for b in pos.to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                // Use upper 32 bits for maximum entropy.
                (h >> 32) as u32
            }
        };

        // Determine which features are needed.
        let any_logprobs = prepared.req_inputs.iter().any(|r| {
            sampling_params_map
                .get(&r.req_id)
                .is_some_and(|p| p.logprobs.is_some())
        });

        let any_grammar = grammar_processor.is_active();
        let any_allowed = allowed_token_ids_processor.is_active();
        let any_seal_pad = seal_pad_processor.is_active();

        let pipeline_active = logits_pipeline.is_some_and(|p| p.any_active());
        let needs_f32 =
            pipeline_active || any_grammar || any_allowed || any_seal_pad || any_logprobs;

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
                    0,
                    input_batch,
                    token_buffers,
                    prompt_lengths,
                );
            }

            // Non-greedy fast path: Gumbel or full sampling on native dtype.
            let all_no_filter = prepared.req_inputs.iter().all(|r| {
                sampling_params_map
                    .get(&r.req_id)
                    .is_none_or(|p| p.top_k <= 0 && p.top_p >= 1.0 && p.min_p <= 0.0)
            });

            let token_ids_owned = if all_no_filter {
                // Pack [temps: f32, seeds: u32] — both 4 bytes, same stride.
                let stride = num_reqs * 4;
                let total_bytes = stride * 2;
                let packed_ptr = Self::get_sampling_packed_ptr_s(host_staging, total_bytes);
                let temps_ptr = packed_ptr as *mut f32;
                let seeds_ptr = unsafe { packed_ptr.add(stride) as *mut u32 };
                for (i, req_slice) in prepared.req_inputs.iter().enumerate() {
                    let params = sampling_params_map.get(&req_slice.req_id);
                    let t = params.map_or(1.0f32, |p| p.temperature.max(1e-7) as f32);
                    unsafe {
                        *temps_ptr.add(i) = t;
                        *seeds_ptr.add(i) = gen_seed(&req_slice.req_id);
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
                let gpu_seeds =
                    unsafe { GpuTensor::new(base.add(stride), &[num_reqs], GpuDType::U32) };
                unsafe {
                    vllm_cuda::kernels::sample_gumbel_batched(
                        logits,
                        gpu_temps,
                        gpu_seeds,
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
                        *randoms_ptr.add(i) =
                            f32::from_bits((gen_seed(&req_slice.req_id) >> 9) | 0x3F800000) - 1.0;
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
                0,
                input_batch,
                token_buffers,
                prompt_lengths,
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

        // 3. Apply grammar mask, allowed_token_ids, and seal-pad on GPU (all need backup logits).
        let needs_mask_backup = (any_grammar && grammar_processor.needs_backup())
            || (any_allowed && allowed_token_ids_processor.needs_backup())
            || (any_seal_pad && seal_pad_processor.needs_backup());
        let _mask_backup_owned = if needs_mask_backup {
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
                .map_err(|e| ExecutorError::WorkerExecution(format!("D2D mask backup: {e}")))?;
                Some(bk)
            };
            let backup = if let Some(raw) = raw_logits_for_logprobs.as_ref() {
                raw.as_gpu_tensor()
            } else {
                backup_owned.as_ref().unwrap().as_gpu_tensor()
            };

            if any_grammar {
                grammar_processor.apply_with_backup(logits_f32, backup, device);
            }
            if any_allowed {
                allowed_token_ids_processor.apply_with_backup(logits_f32, backup, device);
            }
            if any_seal_pad {
                seal_pad_processor.apply_with_backup(logits_f32, backup, device);
            }
            backup_owned
        } else {
            None
        };

        // 4. Apply logit processors pipeline (min_tokens, logit_bias, penalties).
        if let Some(pipeline) = logits_pipeline {
            pipeline.apply_pre_sampling(logits_f32, device);
        }

        // 5. Sample on GPU (from modified f32 logits).
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
                    *randoms_ptr.add(i) =
                        f32::from_bits((gen_seed(&req_slice.req_id) >> 9) | 0x3F800000) - 1.0;
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
            Self::d2h_token_ids_sync(host_staging.as_ref(), 0, &token_ids_gpu, num_reqs, device)?;

        // Build discard mask for intermediate prefill chunks.
        let mut discard = vec![false; num_reqs];
        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
            if req_slice.token_count > 1 {
                let prompt_len = prompt_lengths.get(&req_slice.req_id).copied().unwrap_or(0);
                let tokens_in_pool = input_batch.tokens_in_pool_for(&req_slice.req_id);
                let seq_len_after = tokens_in_pool + req_slice.token_count;
                if seq_len_after < prompt_len {
                    discard[req_idx] = true;
                }
            }
        }

        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
            let tok = host_ids[req_idx];

            // Grammar advance on CPU (Python also does FSM state on CPU).
            #[cfg(feature = "guided-decoding")]
            if !discard[req_idx]
                && let Some(g) = grammar_states.get_mut(&req_slice.req_id)
            {
                g.advance(tok);
            }

            input_batch.commit_step(
                &req_slice.req_id,
                &[tok],
                req_slice.token_count,
                !req_slice.spec_token_ids.is_empty(),
            );
            if !discard[req_idx]
                && let Some(buf) = token_buffers.get_mut(&req_slice.req_id)
            {
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

        // Clear sampled tokens for discarded (intermediate prefill) requests.
        for (idx, d) in discard.iter().enumerate() {
            if *d {
                output.sampled_token_ids[idx].clear();
            }
        }

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

    /// Greedy rejection sampling for speculative decoding.
    ///
    /// When any request has draft tokens, the forward pass produces logits for
    /// ALL token positions `[total_tokens, vocab_size]`. This method:
    /// 1. Argmax all logit rows on GPU → `[total_tokens]` target token IDs
    /// 2. D2H sync all target IDs
    /// 3. CPU greedy rejection: for each spec decode request, compare target
    ///    argmax at each draft position against the draft token. Accept the
    ///    prefix of matches + 1 bonus/recovered token.
    /// 4. Build multi-token ModelRunnerOutput and commit_step.
    ///
    /// Matches Python vLLM's `rejection_sample()` with greedy (no draft probs).
    #[allow(clippy::too_many_arguments)]
    fn spec_decode_greedy_sample(
        logits: GpuTensor,
        total_tokens: usize,
        prepared: PreparedInputs,
        device: &mut GpuDevice,
        _host_staging: &Option<HostStaging>,
        input_batch: &mut InputBatch,
        token_buffers: &mut HashMap<String, Vec<u32>>,
    ) -> ExecutorResult<ModelRunnerOutput> {
        // Sync compute stream to catch any prior kernel errors.
        unsafe { vllm_cuda::driver::stream_synchronize(device.compute_stream) }.map_err(|e| {
            ExecutorError::WorkerExecution(format!("spec decode pre-argmax sync: {e}"))
        })?;

        // Validate logits shape matches total_tokens.
        let logits_dim0 = logits.dim(0);
        assert_eq!(
            logits_dim0,
            total_tokens,
            "spec_decode_greedy_sample: logits dim0 ({}) != total_tokens ({}), num_reqs={}, \
             q_lens={:?}, spec_counts={:?}",
            logits_dim0,
            total_tokens,
            prepared.req_inputs.len(),
            prepared.attn_meta.q_lens,
            prepared
                .req_inputs
                .iter()
                .map(|r| r.spec_token_ids.len())
                .collect::<Vec<_>>(),
        );

        // 1. Argmax all logit rows on GPU.
        let token_ids_owned = unsafe {
            vllm_cuda::kernels::argmax_batched(logits, &mut device.caching, device.compute_stream)
        };
        let token_ids_gpu = token_ids_owned.as_gpu_tensor();

        // Sync compute stream to catch argmax kernel errors before D2H.
        unsafe { vllm_cuda::driver::stream_synchronize(device.compute_stream) }.map_err(|e| {
            ExecutorError::WorkerExecution(format!("spec decode post-argmax sync: {e}"))
        })?;

        // 2. D2H all target argmax IDs.
        // Bypass pinned staging — total_tokens (with drafts) may exceed staging capacity.
        let all_target_ids =
            Self::d2h_token_ids_sync(None, 0, &token_ids_gpu, total_tokens, device)?;

        // 3. Greedy rejection sampling per request.
        let num_reqs = prepared.req_inputs.len();
        let mut req_ids = Vec::with_capacity(num_reqs);
        let mut sampled_token_ids = Vec::with_capacity(num_reqs);
        let mut req_id_to_index = HashMap::with_capacity(num_reqs);

        let mut flat_offset = 0usize;
        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
            let req_id = &req_slice.req_id;
            let n_tokens = req_slice.token_count; // 1 + num_drafts for spec, 1 for normal
            let target_ids = &all_target_ids[flat_offset..flat_offset + n_tokens];

            let rejection =
                crate::input_batch::greedy_rejection_sample(target_ids, &req_slice.spec_token_ids);
            let accepted_tokens = rejection.accepted_tokens;

            // 4. Commit step with accepted tokens.
            input_batch.commit_step(
                req_id,
                &accepted_tokens,
                req_slice.token_count,
                !req_slice.spec_token_ids.is_empty(),
            );
            if let Some(buf) = token_buffers.get_mut(req_id) {
                buf.extend_from_slice(&accepted_tokens);
            }

            req_id_to_index.insert(req_id.clone(), req_idx);
            req_ids.push(req_id.clone());
            sampled_token_ids.push(accepted_tokens);
            flat_offset += n_tokens;
        }

        input_batch.reclaim_buffers(prepared);

        Ok(ModelRunnerOutput {
            req_ids,
            req_id_to_index,
            sampled_token_ids,
            logprobs: None,
            prompt_logprobs_dict: HashMap::new(),
            draft_token_ids: None,
            pooler_output: None,
            d2h_resolver: None,
        })
    }

    /// D2H + sync + commit_step helper (static to avoid borrow conflicts).
    #[allow(clippy::too_many_arguments)]
    fn finalize_d2h_and_commit(
        token_ids_gpu: &GpuTensor,
        num_reqs: usize,
        prepared: PreparedInputs,
        device: &mut GpuDevice,
        host_staging: &Option<HostStaging>,
        buf_idx: usize,
        input_batch: &mut InputBatch,
        token_buffers: &mut HashMap<String, Vec<u32>>,
        prompt_lengths: &HashMap<String, usize>,
    ) -> ExecutorResult<ModelRunnerOutput> {
        let host_ids = Self::d2h_token_ids_sync(
            host_staging.as_ref(),
            buf_idx,
            token_ids_gpu,
            num_reqs,
            device,
        )?;
        // Build discard mask: intermediate prefill chunks (seq_len after
        // this step < prompt_len) should not have their sampled tokens
        // appended to token_buffers. Matches Python's discard_request_mask.
        let mut discard = vec![false; num_reqs];
        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
            if req_slice.token_count > 1 {
                // This is a prefill request. Check if it finishes the prompt.
                let prompt_len = prompt_lengths.get(&req_slice.req_id).copied().unwrap_or(0);
                let tokens_in_pool = input_batch.tokens_in_pool_for(&req_slice.req_id);
                let seq_len_after = tokens_in_pool + req_slice.token_count;
                if seq_len_after < prompt_len {
                    discard[req_idx] = true;
                }
            }
        }

        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
            let tok = host_ids[req_idx];
            input_batch.commit_step(
                &req_slice.req_id,
                &[tok],
                req_slice.token_count,
                !req_slice.spec_token_ids.is_empty(),
            );
            if !discard[req_idx]
                && let Some(buf) = token_buffers.get_mut(&req_slice.req_id)
            {
                buf.push(tok);
            }
        }

        // Build output, clearing tokens for discarded (intermediate prefill) requests.
        let req_ids: Vec<String> = prepared
            .req_inputs
            .iter()
            .map(|r| r.req_id.clone())
            .collect();
        input_batch.reclaim_buffers(prepared);
        let mut output = ModelRunnerOutput::from_ordered(req_ids, host_ids);
        for (idx, d) in discard.iter().enumerate() {
            if *d {
                output.sampled_token_ids[idx].clear();
            }
        }
        Ok(output)
    }
    // -----------------------------------------------------------------------
    // Piecewise CUDA Graph Execution
    // -----------------------------------------------------------------------

    /// Execute a single graph piece during capture.
    unsafe fn execute_graph_piece(
        &self,
        piece_type: GraphPieceType,
        buffers: &vllm_cuda::graph_piece::PersistentBuffers,
        device: &mut GpuDevice,
    ) -> ExecutorResult<()> {
        match piece_type {
            GraphPieceType::Embedding => unsafe { self.execute_embedding_piece(buffers, device) },
            GraphPieceType::LayerPreAttn(layer_idx) => unsafe {
                self.execute_pre_attn_piece(layer_idx, buffers, device)
            },
            GraphPieceType::LayerPostAttn(layer_idx) => unsafe {
                self.execute_post_attn_piece(layer_idx, buffers, device)
            },
            GraphPieceType::LmHead => unsafe { self.execute_lm_head_piece(buffers, device) },
            GraphPieceType::Sampling => unsafe { self.execute_sampling_piece(buffers, device) },
        }
    }

    /// Execute embedding piece: input_ids -> hidden_a
    unsafe fn execute_embedding_piece(
        &self,
        buffers: &vllm_cuda::graph_piece::PersistentBuffers,
        device: &mut GpuDevice,
    ) -> ExecutorResult<()> {
        let batch_size = buffers.max_batch; // Use max_batch during capture
        let input_ids = unsafe { buffers.input_tensors(batch_size).input_ids };

        // Run embedding gather kernel (same as model forward — kernels::embedding_gather)
        let hidden = match &self.model {
            Some(CudaModel::Llama(m)) => unsafe {
                vllm_cuda::kernels::embedding_gather(
                    m.model.embed_tokens.weight,
                    input_ids,
                    &mut device.caching,
                    device.compute_stream,
                )
            },
            Some(CudaModel::Qwen2(m)) => unsafe {
                vllm_cuda::kernels::embedding_gather(
                    m.0.model.embed_tokens.weight,
                    input_ids,
                    &mut device.caching,
                    device.compute_stream,
                )
            },
            Some(CudaModel::Gemma2(m)) => {
                let h = unsafe {
                    vllm_cuda::kernels::embedding_gather(
                        m.model.embed_tokens.weight,
                        input_ids,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };
                // Gemma scaling: hidden *= sqrt(hidden_size)
                unsafe {
                    vllm_cuda::kernels::scale_inplace(
                        h.as_gpu_tensor(),
                        m.model.embed_scale,
                        &device.cublas,
                    )
                };
                h
            }
            Some(CudaModel::Gemma3(m)) => {
                let h = unsafe {
                    vllm_cuda::kernels::embedding_gather(
                        m.model.embed_tokens.weight,
                        input_ids,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };
                unsafe {
                    vllm_cuda::kernels::scale_inplace(
                        h.as_gpu_tensor(),
                        m.model.embed_scale,
                        &device.cublas,
                    )
                };
                h
            }
            Some(CudaModel::Qwen3Moe(m)) => unsafe {
                vllm_cuda::kernels::embedding_gather(
                    m.model.embed_tokens.weight,
                    input_ids,
                    &mut device.caching,
                    device.compute_stream,
                )
            },
            Some(CudaModel::Qwen2Moe(m)) => unsafe {
                vllm_cuda::kernels::embedding_gather(
                    m.model.embed_tokens.weight,
                    input_ids,
                    &mut device.caching,
                    device.compute_stream,
                )
            },
            Some(CudaModel::Mixtral(m)) => unsafe {
                vllm_cuda::kernels::embedding_gather(
                    m.model.embed_tokens.weight,
                    input_ids,
                    &mut device.caching,
                    device.compute_stream,
                )
            },
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "Model not initialized or unsupported".into(),
                ));
            }
        };

        // Copy to persistent buffer hidden_a
        let hidden_bytes = batch_size * buffers.hidden_size * buffers.dtype.size_bytes();
        unsafe {
            driver::memcpy_dtod_async(
                buffers.hidden_a.ptr(),
                hidden.as_gpu_tensor().raw_ptr() as *const u8,
                hidden_bytes,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("Embedding copy: {e}")))?;

        Ok(())
    }

    /// Execute pre-attention piece: hidden -> RMSNorm -> attn_input
    ///
    /// Layer 0: plain rms_norm(hidden_a) → attn_input, copy hidden_a → residual.
    /// Layer N>0: fused_add_rms_norm(hidden_in, residual) → hidden_in becomes normed
    ///            (→ attn_input), residual updated in-place.
    unsafe fn execute_pre_attn_piece(
        &self,
        layer_idx: usize,
        buffers: &vllm_cuda::graph_piece::PersistentBuffers,
        device: &mut GpuDevice,
    ) -> ExecutorResult<()> {
        let batch_size = buffers.max_batch;
        let hidden_bytes = batch_size * buffers.hidden_size * buffers.dtype.size_bytes();

        // Read from hidden_a or hidden_b (ping-pong)
        let use_buffer_a = layer_idx.is_multiple_of(2);
        let hidden_in = unsafe { buffers.hidden_tensor(batch_size, use_buffer_a) };
        let residual = unsafe { buffers.residual_tensor(batch_size) };

        // Get norm weight/eps for the current layer and architecture
        let (norm_weight, norm_eps) = match &self.model {
            Some(CudaModel::Llama(m)) => {
                let layer = &m.model.layers[layer_idx];
                (layer.input_layernorm.weight, layer.input_layernorm.eps)
            }
            Some(CudaModel::Qwen2(m)) => {
                let layer = &m.0.model.layers[layer_idx];
                (layer.input_layernorm.weight, layer.input_layernorm.eps)
            }
            Some(CudaModel::Gemma2(m)) => {
                let layer = &m.model.layers[layer_idx];
                (
                    layer.input_layernorm.inner.weight,
                    layer.input_layernorm.inner.eps,
                )
            }
            Some(CudaModel::Gemma3(m)) => {
                let layer = &m.model.layers[layer_idx];
                (
                    layer.input_layernorm.inner.weight,
                    layer.input_layernorm.inner.eps,
                )
            }
            Some(CudaModel::Qwen3Moe(m)) => {
                let layer = &m.model.layers[layer_idx];
                (layer.input_layernorm.weight, layer.input_layernorm.eps)
            }
            Some(CudaModel::Qwen2Moe(m)) => {
                let layer = &m.model.layers[layer_idx];
                (layer.input_layernorm.weight, layer.input_layernorm.eps)
            }
            Some(CudaModel::Mixtral(m)) => {
                let layer = &m.model.layers[layer_idx];
                (layer.input_layernorm.weight, layer.input_layernorm.eps)
            }
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "Model not initialized or unsupported".into(),
                ));
            }
        };

        if layer_idx == 0 {
            // Layer 0: plain rms_norm, copy hidden → residual
            let normed = unsafe {
                vllm_cuda::kernels::rms_norm(
                    hidden_in,
                    norm_weight,
                    norm_eps,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
            // Copy normed → attn_input
            unsafe {
                driver::memcpy_dtod_async(
                    buffers.attn_input.ptr(),
                    normed.as_gpu_tensor().raw_ptr() as *const u8,
                    hidden_bytes,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("Pre-attn copy: {e}")))?;
            // Copy hidden_in → residual (residual = original hidden before norm)
            unsafe {
                driver::memcpy_dtod_async(
                    buffers.residual.ptr(),
                    hidden_in.raw_ptr() as *const u8,
                    hidden_bytes,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("Pre-attn residual init: {e}")))?;
        } else {
            // Layer N>0: fused_add_rms_norm(hidden_in, residual) → hidden_in = normed, residual updated
            unsafe {
                vllm_cuda::kernels::fused_add_rms_norm_inplace(
                    hidden_in,
                    residual,
                    norm_weight,
                    norm_eps,
                    device.compute_stream,
                );
            }
            // Copy normed hidden_in → attn_input
            unsafe {
                driver::memcpy_dtod_async(
                    buffers.attn_input.ptr(),
                    hidden_in.raw_ptr() as *const u8,
                    hidden_bytes,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("Pre-attn copy: {e}")))?;
        }

        Ok(())
    }

    /// Execute post-attention piece: attn_output + residual -> MLP -> next hidden
    ///
    /// LLaMA/Qwen2: fused_add_rms_norm(attn_out, residual) → normed, then MLP → next hidden.
    /// Gemma2/3: post_attention_layernorm(attn_out), fused_add_rms_norm(normed, residual)
    ///           via pre_feedforward_layernorm, MLP, post_feedforward_layernorm → next hidden.
    unsafe fn execute_post_attn_piece(
        &self,
        layer_idx: usize,
        buffers: &vllm_cuda::graph_piece::PersistentBuffers,
        device: &mut GpuDevice,
    ) -> ExecutorResult<()> {
        let batch_size = buffers.max_batch;
        let attn_out = unsafe { buffers.attn_output_tensor(batch_size) };
        let residual = unsafe { buffers.residual_tensor(batch_size) };

        // Apply post-attention norm + MLP (architecture-specific)
        let mlp_out = match &self.model {
            Some(CudaModel::Llama(m)) => {
                let layer = &m.model.layers[layer_idx];
                unsafe {
                    // Fused add + RMSNorm: attn_out = normed, residual updated
                    vllm_cuda::kernels::fused_add_rms_norm_inplace(
                        attn_out,
                        residual,
                        layer.post_attention_layernorm.weight,
                        layer.post_attention_layernorm.eps,
                        device.compute_stream,
                    );
                    layer.mlp.forward(TensorView::from_raw(attn_out), device)
                }
            }
            Some(CudaModel::Qwen2(m)) => {
                let layer = &m.0.model.layers[layer_idx];
                unsafe {
                    vllm_cuda::kernels::fused_add_rms_norm_inplace(
                        attn_out,
                        residual,
                        layer.post_attention_layernorm.weight,
                        layer.post_attention_layernorm.eps,
                        device.compute_stream,
                    );
                    layer.mlp.forward(TensorView::from_raw(attn_out), device)
                }
            }
            Some(CudaModel::Gemma2(m)) => {
                let layer = &m.model.layers[layer_idx];
                unsafe {
                    // 3. post_attention_layernorm (standalone)
                    let attn_normed = vllm_cuda::kernels::rms_norm(
                        attn_out,
                        layer.post_attention_layernorm.inner.weight,
                        layer.post_attention_layernorm.inner.eps,
                        &mut device.caching,
                        device.compute_stream,
                    );
                    // 4. pre_feedforward_layernorm (fused residual add)
                    vllm_cuda::kernels::fused_add_rms_norm_inplace(
                        *attn_normed,
                        residual,
                        layer.pre_feedforward_layernorm.inner.weight,
                        layer.pre_feedforward_layernorm.inner.eps,
                        device.compute_stream,
                    );
                    // 5. MLP
                    let mlp_out = layer.mlp.forward(attn_normed.view(), device);
                    // 6. post_feedforward_layernorm (standalone)
                    vllm_cuda::kernels::rms_norm(
                        *mlp_out,
                        layer.post_feedforward_layernorm.inner.weight,
                        layer.post_feedforward_layernorm.inner.eps,
                        &mut device.caching,
                        device.compute_stream,
                    )
                }
            }
            Some(CudaModel::Gemma3(m)) => {
                let layer = &m.model.layers[layer_idx];
                unsafe {
                    let attn_normed = vllm_cuda::kernels::rms_norm(
                        attn_out,
                        layer.post_attention_layernorm.inner.weight,
                        layer.post_attention_layernorm.inner.eps,
                        &mut device.caching,
                        device.compute_stream,
                    );
                    vllm_cuda::kernels::fused_add_rms_norm_inplace(
                        *attn_normed,
                        residual,
                        layer.pre_feedforward_layernorm.inner.weight,
                        layer.pre_feedforward_layernorm.inner.eps,
                        device.compute_stream,
                    );
                    let mlp_out = layer.mlp.forward(attn_normed.view(), device);
                    vllm_cuda::kernels::rms_norm(
                        *mlp_out,
                        layer.post_feedforward_layernorm.inner.weight,
                        layer.post_feedforward_layernorm.inner.eps,
                        &mut device.caching,
                        device.compute_stream,
                    )
                }
            }
            Some(CudaModel::Qwen3Moe(m)) => {
                let layer = &m.model.layers[layer_idx];
                unsafe {
                    vllm_cuda::kernels::fused_add_rms_norm_inplace(
                        attn_out,
                        residual,
                        layer.post_attention_layernorm.weight,
                        layer.post_attention_layernorm.eps,
                        device.compute_stream,
                    );
                    layer.mlp.forward(TensorView::from_raw(attn_out), device)
                }
            }
            Some(CudaModel::Qwen2Moe(m)) => {
                let layer = &m.model.layers[layer_idx];
                unsafe {
                    vllm_cuda::kernels::fused_add_rms_norm_inplace(
                        attn_out,
                        residual,
                        layer.post_attention_layernorm.weight,
                        layer.post_attention_layernorm.eps,
                        device.compute_stream,
                    );
                    layer.mlp.forward(TensorView::from_raw(attn_out), device)
                }
            }
            Some(CudaModel::Mixtral(m)) => {
                let layer = &m.model.layers[layer_idx];
                unsafe {
                    vllm_cuda::kernels::fused_add_rms_norm_inplace(
                        attn_out,
                        residual,
                        layer.post_attention_layernorm.weight,
                        layer.post_attention_layernorm.eps,
                        device.compute_stream,
                    );
                    layer
                        .block_sparse_moe
                        .forward(TensorView::from_raw(attn_out), device)
                }
            }
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "Model not initialized or unsupported".into(),
                ));
            }
        };

        // Copy to next buffer (ping-pong)
        let use_buffer_a = layer_idx.is_multiple_of(2);
        let next_buffer = if use_buffer_a {
            buffers.hidden_b.ptr()
        } else {
            buffers.hidden_a.ptr()
        };
        let hidden_bytes = batch_size * buffers.hidden_size * buffers.dtype.size_bytes();
        unsafe {
            driver::memcpy_dtod_async(
                next_buffer,
                mlp_out.as_gpu_tensor().raw_ptr() as *const u8,
                hidden_bytes,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("Post-attn copy: {e}")))?;

        Ok(())
    }

    /// Execute LM head piece: final hidden -> logits
    ///
    /// The final hidden state needs a fused_add_rms_norm with the residual (same as
    /// pre-attn for layer N>0), then the final norm, then the LM head projection.
    unsafe fn execute_lm_head_piece(
        &self,
        buffers: &vllm_cuda::graph_piece::PersistentBuffers,
        device: &mut GpuDevice,
    ) -> ExecutorResult<()> {
        let batch_size = buffers.max_batch;
        let num_layers = buffers.num_layers;

        // Read from final hidden buffer (after last post-attn piece wrote here)
        let use_buffer_a = num_layers.is_multiple_of(2);
        let hidden = unsafe { buffers.hidden_tensor(batch_size, use_buffer_a) };
        let residual = unsafe { buffers.residual_tensor(batch_size) };

        // Apply final norm + LM head
        let logits = match &self.model {
            Some(CudaModel::Llama(m)) => {
                unsafe {
                    // Fused add of last MLP output into residual, then final norm
                    vllm_cuda::kernels::fused_add_rms_norm_inplace(
                        hidden,
                        residual,
                        m.model.norm.weight,
                        m.model.norm.eps,
                        device.compute_stream,
                    );
                    // hidden is now normed; project to vocab
                    m.lm_head.forward(
                        TensorView::from_raw(hidden),
                        &mut device.cublas,
                        &mut device.caching,
                        device.compute_stream,
                    )
                }
            }
            Some(CudaModel::Qwen2(m)) => unsafe {
                vllm_cuda::kernels::fused_add_rms_norm_inplace(
                    hidden,
                    residual,
                    m.0.model.norm.weight,
                    m.0.model.norm.eps,
                    device.compute_stream,
                );
                m.0.lm_head.forward(
                    TensorView::from_raw(hidden),
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                )
            },
            Some(CudaModel::Gemma2(m)) => {
                unsafe {
                    // Gemma2: last layer output is post_feedforward_normed.
                    // Need fused add into residual + final model norm.
                    vllm_cuda::kernels::fused_add_rms_norm_inplace(
                        hidden,
                        residual,
                        m.model.norm.inner.weight,
                        m.model.norm.inner.eps,
                        device.compute_stream,
                    );
                    m.lm_head.forward(
                        TensorView::from_raw(hidden),
                        &mut device.cublas,
                        &mut device.caching,
                    )
                }
            }
            Some(CudaModel::Gemma3(m)) => unsafe {
                vllm_cuda::kernels::fused_add_rms_norm_inplace(
                    hidden,
                    residual,
                    m.model.norm.inner.weight,
                    m.model.norm.inner.eps,
                    device.compute_stream,
                );
                m.lm_head.forward(
                    TensorView::from_raw(hidden),
                    &mut device.cublas,
                    &mut device.caching,
                )
            },
            Some(CudaModel::Qwen3Moe(m)) => unsafe {
                vllm_cuda::kernels::fused_add_rms_norm_inplace(
                    hidden,
                    residual,
                    m.model.norm.weight,
                    m.model.norm.eps,
                    device.compute_stream,
                );
                m.lm_head.forward(
                    TensorView::from_raw(hidden),
                    &mut device.cublas,
                    &mut device.caching,
                )
            },
            Some(CudaModel::Qwen2Moe(m)) => unsafe {
                vllm_cuda::kernels::fused_add_rms_norm_inplace(
                    hidden,
                    residual,
                    m.model.norm.weight,
                    m.model.norm.eps,
                    device.compute_stream,
                );
                m.lm_head.forward(
                    TensorView::from_raw(hidden),
                    &mut device.cublas,
                    &mut device.caching,
                )
            },
            Some(CudaModel::Mixtral(m)) => unsafe {
                vllm_cuda::kernels::fused_add_rms_norm_inplace(
                    hidden,
                    residual,
                    m.model.norm.weight,
                    m.model.norm.eps,
                    device.compute_stream,
                );
                m.lm_head.forward(
                    TensorView::from_raw(hidden),
                    &mut device.cublas,
                    &mut device.caching,
                )
            },
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "Model not initialized or unsupported".into(),
                ));
            }
        };

        // Copy to logits buffer
        let logits_bytes = batch_size * buffers.vocab_size * buffers.dtype.size_bytes();
        unsafe {
            driver::memcpy_dtod_async(
                buffers.logits.ptr(),
                logits.as_gpu_tensor().raw_ptr() as *const u8,
                logits_bytes,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("LM head copy: {e}")))?;

        Ok(())
    }

    /// Execute sampling piece: logits -> token_ids
    unsafe fn execute_sampling_piece(
        &self,
        buffers: &vllm_cuda::graph_piece::PersistentBuffers,
        device: &mut GpuDevice,
    ) -> ExecutorResult<()> {
        let batch_size = buffers.max_batch;
        let logits = unsafe { buffers.logits_tensor(batch_size) };

        // Argmax sampling
        let token_ids = unsafe {
            vllm_cuda::kernels::argmax_batched(logits, &mut device.caching, device.compute_stream)
        };

        // Copy to token_ids buffer
        unsafe {
            driver::memcpy_dtod_async(
                buffers.token_ids.ptr(),
                token_ids.as_gpu_tensor().raw_ptr() as *const u8,
                batch_size * 4,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("Sampling copy: {e}")))?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Piecewise Graph Capture Helper
    // -----------------------------------------------------------------------

    /// Capture piecewise CUDA graphs for a set of batch sizes.
    /// Called from compile_or_warm_up_model after releasing the model/kv_cache/device borrows.
    fn capture_piecewise_graphs(
        &mut self,
        num_layers: usize,
        hidden_size: usize,
        vocab_size: usize,
        dtype: vllm_cuda::dtype::DType,
        max_bs: usize,
        capture_sizes: &[usize],
    ) {
        info!("Capturing piecewise CUDA graphs (attention excluded from graphs)...");
        match unsafe {
            PiecewiseGraphRunner::new(max_bs, hidden_size, vocab_size, num_layers, dtype)
        } {
            Ok(mut piecewise_runner) => {
                // SAFETY: execute_graph_piece reads self.model (shared) while
                // capture_all_pieces mutably borrows self.device. These are disjoint
                // fields, but the borrow checker can't verify that, so we use a raw
                // pointer for the shared model access.
                let self_ptr: *const Self = self;

                for &bs in capture_sizes.iter().rev() {
                    info!("Capturing piecewise graphs for batch_size={bs}...");
                    let dev = self.device.as_mut().unwrap();
                    // Fill buffers with valid dummy data before capture warmup.
                    if let Err(e) = unsafe { piecewise_runner.fill_dummy_decode(bs, dev) } {
                        tracing::warn!("Failed to fill dummy decode data for bs={bs}: {e}");
                        break;
                    }
                    let result = unsafe {
                        // Suppress NCCL during capture so collectives are NOT
                        // baked into graphs. They run eagerly during replay.
                        #[cfg(feature = "nccl")]
                        let _nccl_guard = vllm_cuda::nccl::SuppressNcclGuard::new();
                        piecewise_runner.capture_all_pieces(bs, dev, |piece_type, buffers, d| {
                            (*self_ptr)
                                .execute_graph_piece(piece_type, buffers, d)
                                .map_err(|e| anyhow::anyhow!("{e}"))
                        })
                    };
                    match result {
                        Ok(()) => info!("Piecewise graphs captured for batch_size={bs}"),
                        Err(e) => {
                            tracing::warn!("Failed to capture piecewise graphs for bs={bs}: {e}");
                            break;
                        }
                    }
                }

                if !piecewise_runner.captured_sizes().is_empty() {
                    info!(
                        "Piecewise CUDA graphs captured for batch sizes: {:?}",
                        piecewise_runner.captured_sizes()
                    );
                    self.piecewise_graph_runner = Some(piecewise_runner);
                } else {
                    tracing::warn!(
                        "No piecewise graphs captured, piecewise mode will be unavailable"
                    );
                }
            }
            Err(e) => {
                tracing::warn!("Failed to create PiecewiseGraphRunner: {e}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Dynamic Attention Execution (Outside CUDA Graphs)
    // -----------------------------------------------------------------------

    /// Execute attention for a single layer. Called during piecewise replay
    /// between pre-attn and post-attn graph pieces.
    ///
    /// FA2 handles split-K internally — no manual split-K optimization needed.
    #[allow(clippy::too_many_arguments)]
    unsafe fn execute_attention_layer(
        &self,
        layer_idx: usize,
        attn_input: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
    ) -> ExecutorResult<OwnedTensor> {
        match &self.model {
            Some(CudaModel::Llama(m)) => {
                let layer = &m.model.layers[layer_idx];
                let rotary = &m.model.rotary;
                Ok(unsafe {
                    layer.self_attn.forward(
                        TensorView::from_raw(attn_input),
                        TensorView::from_raw(positions),
                        TensorView::from_raw(slot_mapping),
                        TensorView::from_raw(cu_seqlens_q),
                        TensorView::from_raw(seqused_k),
                        TensorView::from_raw(block_table),
                        max_seqlen_q,
                        max_seqlen_k,
                        kv_cache,
                        rotary,
                        device,
                    )
                })
            }
            Some(CudaModel::Qwen2(m)) => {
                let layer = &m.0.model.layers[layer_idx];
                let rotary = &m.0.model.rotary;
                Ok(unsafe {
                    layer.self_attn.forward(
                        TensorView::from_raw(attn_input),
                        TensorView::from_raw(positions),
                        TensorView::from_raw(slot_mapping),
                        TensorView::from_raw(cu_seqlens_q),
                        TensorView::from_raw(seqused_k),
                        TensorView::from_raw(block_table),
                        max_seqlen_q,
                        max_seqlen_k,
                        kv_cache,
                        rotary,
                        device,
                    )
                })
            }
            Some(CudaModel::Gemma2(m)) => {
                let layer = &m.model.layers[layer_idx];
                let rotary = &m.model.rotary;
                Ok(unsafe {
                    layer.self_attn.forward(
                        TensorView::from_raw(attn_input),
                        TensorView::from_raw(positions),
                        TensorView::from_raw(slot_mapping),
                        TensorView::from_raw(cu_seqlens_q),
                        TensorView::from_raw(seqused_k),
                        TensorView::from_raw(block_table),
                        max_seqlen_q,
                        max_seqlen_k,
                        kv_cache,
                        rotary,
                        device,
                    )
                })
            }
            Some(CudaModel::Gemma3(m)) => {
                let layer = &m.model.layers[layer_idx];
                let is_sliding = layer_idx < m.model.layer_is_sliding.len()
                    && m.model.layer_is_sliding[layer_idx];
                let rotary = if is_sliding {
                    &m.model.rotary_local
                } else {
                    &m.model.rotary_global
                };
                Ok(unsafe {
                    layer.self_attn.forward(
                        TensorView::from_raw(attn_input),
                        TensorView::from_raw(positions),
                        TensorView::from_raw(slot_mapping),
                        TensorView::from_raw(cu_seqlens_q),
                        TensorView::from_raw(seqused_k),
                        TensorView::from_raw(block_table),
                        max_seqlen_q,
                        max_seqlen_k,
                        kv_cache,
                        rotary,
                        device,
                    )
                })
            }
            Some(CudaModel::Qwen3Moe(m)) => {
                let layer = &m.model.layers[layer_idx];
                let rotary = &m.model.rotary;
                Ok(unsafe {
                    layer.self_attn.forward(
                        TensorView::from_raw(attn_input),
                        TensorView::from_raw(positions),
                        TensorView::from_raw(slot_mapping),
                        TensorView::from_raw(cu_seqlens_q),
                        TensorView::from_raw(seqused_k),
                        TensorView::from_raw(block_table),
                        max_seqlen_q,
                        max_seqlen_k,
                        kv_cache,
                        rotary,
                        device,
                    )
                })
            }
            Some(CudaModel::Qwen2Moe(m)) => {
                let layer = &m.model.layers[layer_idx];
                let rotary = &m.model.rotary;
                Ok(unsafe {
                    layer.self_attn.forward(
                        TensorView::from_raw(attn_input),
                        TensorView::from_raw(positions),
                        TensorView::from_raw(slot_mapping),
                        TensorView::from_raw(cu_seqlens_q),
                        TensorView::from_raw(seqused_k),
                        TensorView::from_raw(block_table),
                        max_seqlen_q,
                        max_seqlen_k,
                        kv_cache,
                        rotary,
                        device,
                    )
                })
            }
            Some(CudaModel::Mixtral(m)) => {
                let layer = &m.model.layers[layer_idx];
                let rotary = &m.model.rotary;
                Ok(unsafe {
                    layer.self_attn.forward(
                        TensorView::from_raw(attn_input),
                        TensorView::from_raw(positions),
                        TensorView::from_raw(slot_mapping),
                        TensorView::from_raw(cu_seqlens_q),
                        TensorView::from_raw(seqused_k),
                        TensorView::from_raw(block_table),
                        max_seqlen_q,
                        max_seqlen_k,
                        kv_cache,
                        rotary,
                        device,
                    )
                })
            }
            _ => Err(ExecutorError::WorkerExecution(
                "Model not initialized or unsupported for attention".into(),
            )),
        }
    }

    // -----------------------------------------------------------------------
    // Piecewise Replay Flow
    // -----------------------------------------------------------------------

    /// Copy input metadata to persistent buffers for piecewise execution.
    ///
    /// `PreparedInputs` holds CPU-side data. We H2D-copy into the persistent GPU
    /// buffers whose pointers are baked into the captured graphs.
    /// `graph_bs` is the padded batch size matching the captured graph.
    unsafe fn copy_inputs_to_buffers(
        prepared: &PreparedInputs,
        buffers: &vllm_cuda::graph_piece::PersistentBuffers,
        block_size: usize,
        graph_bs: usize,
        device: &mut GpuDevice,
    ) -> ExecutorResult<()> {
        let meta = &prepared.attn_meta;
        let batch_size = meta.num_reqs;

        // ---- input_ids (u32, decode: 1 token per request) ----
        // Pad to graph_bs with zeros (matching Python's persistent buffer approach).
        {
            let mut ids: Vec<u32> = prepared.flat_token_ids.to_vec();
            ids.resize(graph_bs, 0);
            unsafe {
                driver::memcpy_htod_async(
                    buffers.input_ids.ptr(),
                    ids.as_ptr() as *const u8,
                    graph_bs * 4,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("Copy input_ids: {e}")))?;
        }

        // ---- positions (u32) — last token position per request ----
        {
            let mut positions: Vec<u32> = prepared.flat_positions.to_vec();
            positions.resize(graph_bs, 0);
            unsafe {
                driver::memcpy_htod_async(
                    buffers.positions.ptr(),
                    positions.as_ptr() as *const u8,
                    graph_bs * 4,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("Copy positions: {e}")))?;
        }

        // ---- slot_mapping (i64) ----
        {
            let mut slot_mapping: Vec<i64> = Vec::with_capacity(graph_bs);
            for i in 0..batch_size {
                let tokens_before = meta.tokens_before[i];
                let q_len = meta.q_lens[i];
                let block_ids = &meta.block_ids[i];
                let abs_pos = tokens_before + q_len - 1;
                let block_idx = abs_pos / block_size;
                let offset = abs_pos % block_size;
                if block_idx < block_ids.len() {
                    slot_mapping.push((block_ids[block_idx] * block_size + offset) as i64);
                } else {
                    slot_mapping.push(-1i64);
                }
            }
            // Pad with -1 (no write) for extra slots
            slot_mapping.resize(graph_bs, -1i64);
            unsafe {
                driver::memcpy_htod_async(
                    buffers.slot_mapping.ptr(),
                    slot_mapping.as_ptr() as *const u8,
                    graph_bs * 8,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("Copy slot_mapping: {e}")))?;
        }

        // ---- cu_seqlens_q (i32, length graph_bs+1) ----
        // For decode: [0, 1, 2, ..., graph_bs]. Padded entries get q_len=1.
        {
            let mut cu_q: Vec<i32> = meta.query_start_loc.iter().map(|&x| x as i32).collect();
            // Extend to graph_bs + 1 entries (each padded request has q_len=1)
            let last = *cu_q.last().unwrap_or(&0);
            for i in 0..(graph_bs - batch_size) {
                cu_q.push(last + (i as i32) + 1);
            }
            unsafe {
                driver::memcpy_htod_async(
                    buffers.cu_seqlens_q.ptr(),
                    cu_q.as_ptr() as *const u8,
                    (graph_bs + 1) * 4,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("Copy cu_seqlens_q: {e}")))?;
        }

        // ---- seqused_k (i32) ----
        // Padded entries get seqused_k=1 (minimal valid KV; reads from block 0).
        {
            let mut seqused: Vec<i32> = meta.seq_lens.iter().map(|&x| x as i32).collect();
            seqused.resize(graph_bs, 1);
            unsafe {
                driver::memcpy_htod_async(
                    buffers.seqused_k.ptr(),
                    seqused.as_ptr() as *const u8,
                    graph_bs * 4,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("Copy seqused_k: {e}")))?;
        }

        // ---- block_table (i32, [graph_bs, max_blocks_per_seq]) ----
        {
            let max_blocks = buffers.max_blocks_per_seq;
            let mut block_table = vec![0i32; graph_bs * max_blocks];
            for (i, blocks) in meta.block_ids.iter().enumerate() {
                for (j, &bid) in blocks.iter().enumerate() {
                    if j < max_blocks {
                        block_table[i * max_blocks + j] = bid as i32;
                    }
                }
            }
            // Padded entries: block_table stays 0 (block 0 is valid memory).
            unsafe {
                driver::memcpy_htod_async(
                    buffers.block_table.ptr(),
                    block_table.as_ptr() as *const u8,
                    graph_bs * max_blocks * 4,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("Copy block_table: {e}")))?;
        }

        Ok(())
    }

    /// Execute model using piecewise CUDA graphs with dynamic attention.
    /// This is the main replay flow that orchestrates piece execution.
    unsafe fn execute_model_piecewise(
        &mut self,
        prepared: &PreparedInputs,
    ) -> ExecutorResult<Vec<u32>> {
        // ---- Extract all read-only metadata we need before any &mut borrows ----
        let actual_batch_size = prepared.attn_meta.num_reqs;
        let max_seqlen_q = prepared.attn_meta.q_lens.iter().copied().max().unwrap_or(1);
        let max_seqlen_k = prepared
            .attn_meta
            .seq_lens
            .iter()
            .copied()
            .max()
            .unwrap_or(1);

        // Check runner exists and find the nearest captured graph size.
        let (max_batch, num_layers, hidden_size, dtype_size, attn_output_ptr, batch_size) = {
            let runner = self.piecewise_graph_runner.as_ref().ok_or_else(|| {
                ExecutorError::WorkerExecution("Piecewise graph runner not initialized".into())
            })?;
            let b = &runner.buffers;
            let graph_bs = runner
                .nearest_graph_size(actual_batch_size)
                .ok_or_else(|| {
                    ExecutorError::WorkerExecution(format!(
                        "No piecewise graph for batch_size={}",
                        actual_batch_size
                    ))
                })?;
            (
                b.max_batch,
                b.num_layers,
                b.hidden_size,
                b.dtype.size_bytes(),
                b.attn_output.ptr(),
                graph_bs,
            )
        };

        if batch_size > max_batch {
            return Err(ExecutorError::WorkerExecution(format!(
                "Batch size {} exceeds max_batch {}",
                batch_size, max_batch
            )));
        }

        let kv_cache_block_size = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerExecution("KV cache not initialized".into()))?
            .block_size;

        // ---- H2D copy inputs into persistent buffers ----
        {
            let device = self
                .device
                .as_mut()
                .ok_or_else(|| ExecutorError::WorkerExecution("Device not initialized".into()))?;
            let runner = self.piecewise_graph_runner.as_ref().unwrap();
            unsafe {
                Self::copy_inputs_to_buffers(
                    prepared,
                    &runner.buffers,
                    kv_cache_block_size,
                    batch_size,
                    device,
                )?;
            }
        }

        // ---- Replay embedding piece ----
        {
            let device = self.device.as_mut().unwrap();
            let runner = self.piecewise_graph_runner.as_ref().unwrap();
            unsafe {
                runner
                    .replay_piece(batch_size, GraphPieceType::Embedding, device)
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("Embedding replay: {e}"))
                    })?;
            }
        }

        // ---- Execute all layers with dynamic attention ----
        for layer_idx in 0..num_layers {
            // Replay pre-attention piece (RMSNorm → attn_input buffer).
            {
                let device = self.device.as_mut().unwrap();
                let runner = self.piecewise_graph_runner.as_ref().unwrap();
                unsafe {
                    runner
                        .replay_piece(batch_size, GraphPieceType::LayerPreAttn(layer_idx), device)
                        .map_err(|e| {
                            ExecutorError::WorkerExecution(format!(
                                "Layer {} pre-attn replay: {e}",
                                layer_idx
                            ))
                        })?;
                }
            }

            // Build GpuTensor views into persistent buffers for attention inputs.
            // SAFETY: pointers are valid for the lifetime of PersistentBuffers.
            let (
                attn_input,
                positions_t,
                slot_mapping_t,
                cu_seqlens_q_t,
                seqused_k_t,
                block_table_t,
            ) = {
                let runner = self.piecewise_graph_runner.as_ref().unwrap();
                let b = &runner.buffers;
                unsafe {
                    (
                        b.attn_input_tensor(batch_size),
                        b.positions_tensor(batch_size),
                        b.slot_mapping_tensor(batch_size),
                        b.cu_seqlens_q_tensor(batch_size),
                        b.seqused_k_tensor(batch_size),
                        b.block_table_tensor(batch_size),
                    )
                }
            };

            // Execute attention dynamically (NOT captured in graph).
            // SAFETY: device, kv_cache, and model are separate fields of self;
            // we use raw pointers to satisfy the borrow checker while keeping
            // all three accessible simultaneously.
            let attn_output = unsafe {
                let self_ptr: *const CudaWorker = self;
                let device_ptr: *mut GpuDevice = self.device.as_mut().unwrap();
                let kv_cache_ptr: *const KvCachePool = self.kv_cache.as_ref().unwrap();
                (*self_ptr).execute_attention_layer(
                    layer_idx,
                    attn_input,
                    positions_t,
                    slot_mapping_t,
                    cu_seqlens_q_t,
                    seqused_k_t,
                    block_table_t,
                    max_seqlen_q,
                    max_seqlen_k,
                    &*kv_cache_ptr,
                    &mut *device_ptr,
                )?
            };

            // Copy attention output into the persistent attn_output buffer.
            {
                let device = self.device.as_mut().unwrap();
                let attn_bytes = batch_size * hidden_size * dtype_size;
                unsafe {
                    driver::memcpy_dtod_async(
                        attn_output_ptr,
                        attn_output.as_gpu_tensor().raw_ptr() as *const u8,
                        attn_bytes,
                        device.compute_stream,
                    )
                }
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!(
                        "Layer {} attn output copy: {e}",
                        layer_idx
                    ))
                })?;
            }

            // Replay post-attention piece (residual add + MLP → next hidden buffer).
            // NCCL all-reduce is suppressed inside the graph; run it eagerly after.
            {
                let device = self.device.as_mut().unwrap();
                let runner = self.piecewise_graph_runner.as_ref().unwrap();
                unsafe {
                    #[cfg(feature = "nccl")]
                    let _nccl_guard = vllm_cuda::nccl::SuppressNcclGuard::new();
                    runner
                        .replay_piece(batch_size, GraphPieceType::LayerPostAttn(layer_idx), device)
                        .map_err(|e| {
                            ExecutorError::WorkerExecution(format!(
                                "Layer {} post-attn replay: {e}",
                                layer_idx
                            ))
                        })?;
                }
            }

            // Eager MLP all-reduce (TP): the PostAttn graph wrote the down_proj
            // output (before all-reduce) into the next hidden buffer.
            #[cfg(feature = "nccl")]
            {
                let use_buffer_a = (layer_idx + 1).is_multiple_of(2);
                let runner = self.piecewise_graph_runner.as_ref().unwrap();
                let hidden_buf = unsafe { runner.buffers.hidden_tensor(batch_size, use_buffer_a) };
                match &self.model {
                    Some(CudaModel::Llama(m)) => {
                        if let Some(ref group) = m.model.layers[layer_idx].mlp.tp_group {
                            unsafe {
                                group
                                    .all_reduce_inplace(hidden_buf)
                                    .expect("piecewise MLP all_reduce failed");
                            }
                        }
                    }
                    Some(CudaModel::Qwen2(m)) => {
                        if let Some(ref group) = m.0.model.layers[layer_idx].mlp.tp_group {
                            unsafe {
                                group
                                    .all_reduce_inplace(hidden_buf)
                                    .expect("piecewise MLP all_reduce failed");
                            }
                        }
                    }
                    _ => {} // Other models: no TP all-reduce needed
                }
            }
        }

        // ---- Replay LM head piece ----
        {
            let device = self.device.as_mut().unwrap();
            let runner = self.piecewise_graph_runner.as_ref().unwrap();
            unsafe {
                #[cfg(feature = "nccl")]
                let _nccl_guard = vllm_cuda::nccl::SuppressNcclGuard::new();
                runner
                    .replay_piece(batch_size, GraphPieceType::LmHead, device)
                    .map_err(|e| ExecutorError::WorkerExecution(format!("LM head replay: {e}")))?;
            }
        }

        // Eager all-gather logits (TP): the LmHead graph produced a vocab shard.
        // Rearrange into full vocab logits in the persistent logits buffer.
        #[cfg(feature = "nccl")]
        {
            let tp_group = match &self.model {
                Some(CudaModel::Llama(m)) => m.tp_group.as_ref(),
                Some(CudaModel::Qwen2(m)) => m.0.tp_group.as_ref(),
                _ => None,
            };
            if let Some(group) = tp_group {
                let runner = self.piecewise_graph_runner.as_ref().unwrap();
                let device = self.device.as_mut().unwrap();
                let logits_shard = unsafe { runner.buffers.logits_tensor(batch_size) };
                let gathered =
                    unsafe { group.all_gather_last_dim(logits_shard, &mut device.caching) };
                // Copy gathered logits back into the persistent logits buffer.
                let gathered_bytes = gathered.as_gpu_tensor().numel()
                    * gathered.as_gpu_tensor().dtype().size_bytes();
                unsafe {
                    vllm_cuda::driver::memcpy_dtod_async(
                        runner.buffers.logits.ptr(),
                        gathered.as_gpu_tensor().raw_ptr() as *const u8,
                        gathered_bytes,
                        device.compute_stream,
                    )
                }
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("logits all-gather copy: {e}"))
                })?;
            }
        }

        // ---- Replay sampling piece ----
        {
            let device = self.device.as_mut().unwrap();
            let runner = self.piecewise_graph_runner.as_ref().unwrap();
            unsafe {
                runner
                    .replay_piece(batch_size, GraphPieceType::Sampling, device)
                    .map_err(|e| ExecutorError::WorkerExecution(format!("Sampling replay: {e}")))?;
            }
        }

        // ---- Extract output token IDs from persistent buffer via D2H copy ----
        // Only copy actual_batch_size tokens (ignore padded slots).
        let token_ids_gpu = {
            let runner = self.piecewise_graph_runner.as_ref().unwrap();
            let buffers = &runner.buffers;
            // SAFETY: token_ids buffer is valid and was written by the sampling piece.
            unsafe { GpuTensor::new(buffers.token_ids.ptr(), &[actual_batch_size], GpuDType::U32) }
        };
        let staging = self.host_staging.as_ref();
        let device = self.device.as_mut().unwrap();
        Self::d2h_token_ids_sync(staging, 0, &token_ids_gpu, actual_batch_size, device)
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

        // Ensure CUDA context is current on this thread.
        {
            let device = self
                .device
                .as_ref()
                .ok_or_else(|| ExecutorError::WorkerInit("device not initialized".into()))?;
            unsafe { driver::ctx_set_current(device.ctx) }
                .map_err(|e| ExecutorError::WorkerInit(format!("ctx_set_current: {e}")))?;
        }

        // 1. Resolve model directory (or GGUF file path).
        let model_dir = self.resolve_model_path()?;
        info!("CudaWorker: loading model from {}", model_dir.display());

        let device = self
            .device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("device not initialized".into()))?;

        // Tokenizer on background thread. Sources tried, in order:
        //   1. `tokenizer.json` in the dir (parent dir if model_dir
        //      is a bare `.gguf` file path).
        //   2. GGUF metadata `tokenizer.ggml.*` if a `.gguf` is found
        //      (covers HF GGUF repos that don't ship a sibling
        //      tokenizer.json — unsloth, bartowski, etc.).
        let tok_search_dir = if model_dir.is_file() {
            model_dir.parent().unwrap_or(&model_dir).to_path_buf()
        } else {
            model_dir.clone()
        };
        let tok_gguf_path: Option<PathBuf> =
            if model_dir.is_file() && model_dir.extension().is_some_and(|e| e == "gguf") {
                Some(model_dir.clone())
            } else {
                std::fs::read_dir(&tok_search_dir).ok().and_then(|mut it| {
                    it.find_map(|entry| {
                        let p = entry.ok()?.path();
                        (p.extension().is_some_and(|e| e == "gguf")).then_some(p)
                    })
                })
            };
        let tokenizer_handle = std::thread::spawn(move || {
            let json_path = tok_search_dir.join("tokenizer.json");
            if json_path.exists()
                && let Ok(t) = tokenizers::Tokenizer::from_file(&json_path)
            {
                return Some(t);
            }
            if let Some(gguf_path) = tok_gguf_path
                && let Ok(gguf) = ferrite_gguf::GgufFile::open(&gguf_path)
            {
                match ferrite_gguf::gguf_tokenizer(&gguf) {
                    Ok(Some(t)) => return Some(t),
                    Ok(None) => {}
                    Err(e) => tracing::warn!("gguf_tokenizer failed: {e}"),
                }
            }
            None
        });

        // 2. Parse config: GGUF metadata for `.gguf` files, otherwise
        //    `config.json` in the model dir. GGUF support lives in
        //    `ferrite-gguf` — vllm-model has no GGUF-awareness.
        let hf_config = if model_dir.is_file() && model_dir.extension().is_some_and(|e| e == "gguf")
        {
            let gguf = ferrite_gguf::GgufFile::open(&model_dir)
                .map_err(|e| ExecutorError::WorkerInit(format!("GGUF parse failed: {e}")))?;
            ferrite_gguf::gguf_model_config(&gguf)
                .map_err(|e| ExecutorError::WorkerInit(format!("GGUF config: {e}")))?
        } else {
            HfModelConfig::from_path(&model_dir)
                .map_err(|e| ExecutorError::WorkerInit(format!("config parse failed: {e}")))?
        };

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

        // 5. Parse weight files. Safetensors stays CPU-mmap'd (lazy
        // upload via take()); GGUF is eagerly uploaded via the
        // inventory-registered loader. `from_path` dispatches.
        let tp_world = self.config.tp_world_size.max(1);
        let tp_rank = self.config.tp_rank;
        let stream = device.compute_stream;
        let mut weights = unsafe {
            let device_mut = self.device.as_mut().unwrap();
            GpuWeights::from_path(
                &model_dir,
                stream,
                &mut device_mut.caching,
                dtype,
                tp_rank,
                tp_world,
            )
        }
        .map_err(|e| ExecutorError::WorkerInit(format!("weight load failed: {e}")))?;
        info!("CudaWorker: parsed {} weight tensors", weights.len());
        let uses_ggml = weights.is_gguf();
        let device = self.device.as_ref().unwrap();

        // Set target dtype so F32 weights are cast to model dtype on load.
        // Matches Python vLLM where model parameters are initialized with
        // torch_dtype and PyTorch auto-casts during weight_loader copy.
        weights.set_target_dtype(dtype);

        // 5b. Merge LoRA adapter weights (CPU-side, before H2D copy).
        if let Some(ref adapter_path) = self.config.lora_adapter {
            let adapter_dir = self.resolve_adapter_path(adapter_path)?;
            let merged = weights
                .merge_lora(&adapter_dir)
                .map_err(|e| ExecutorError::WorkerInit(format!("LoRA merge: {e}")))?;
            info!("CudaWorker: merged {merged} LoRA weight tensors");
        }

        // 6. Detect quantization config.
        let qconfig = quant::detect_quant_config(&model_dir)
            .map_err(|e| ExecutorError::WorkerInit(format!("quant config detection: {e}")))?;
        if qconfig.is_quantized() {
            info!("CudaWorker: detected quantization: {:?}", qconfig);
            if self.config.lora_adapter.is_some() {
                return Err(ExecutorError::WorkerInit(
                    "LoRA with quantized models requires Punica kernels (not yet implemented)"
                        .into(),
                ));
            }
        }

        // 6b. Start background pre-cast pipeline. This pre-faults mmap pages
        // and casts float tensors into pinned buffers concurrently with model
        // construction. Must be after set_target_dtype() and merge_lora().
        weights.start_precast();

        // 7. Construct model based on architecture.
        let tp_world = self.config.tp_world_size;
        let tp_rank = self.config.tp_rank;
        let use_tp = tp_world > 1;
        let tp = vllm_cuda::model::llama::TpConfig {
            rank: tp_rank,
            world_size: tp_world,
        };
        let use_pp = self.config.pp_size > 1;
        let pp_config = if use_pp {
            let num_layers = hf_config.num_hidden_layers.unwrap_or(1);
            Some(vllm_cuda::PpConfig::new(
                num_layers,
                self.config.pp_rank,
                self.config.pp_size,
            ))
        } else {
            None
        };

        // ── Try ferrite first (all supported arches in one shot).
        //
        // `ferrite_forward::try_load` walks every `#[forward]`-
        // registered arch; first whose `hf_arches` list contains
        // `arch.as_str()` wins. Auto-registered via
        // `inventory::submit!` at macro expansion — adding a new
        // arch to ferrite-models touches zero lines here.
        let disable_ferrite = std::env::var("FERRITE_DISABLE").ok().as_deref() == Some("1");
        let ferrite_eligible = (!qconfig.is_quantized()
            || qconfig.is_awq()
            || qconfig.is_gptq()
            || qconfig.is_bnb4bit()
            || qconfig.is_fp8())
            && !use_pp
            && !disable_ferrite;
        let ferrite_loaded: Option<CudaModel> = if ferrite_eligible {
            let stream = device.compute_stream;
            // Thread a minimal HF-config view into ferrite so per-
            // variant `fingerprint_matches` can disambiguate
            // checkpoints that share tensor shapes but differ in
            // config-only fields (Phi-3-mini-4k vs Phi-3.5-mini-128k
            // — same weights, different `max_position_embeddings` +
            // `rope_scaling.type`). Ferrite owns the rest.
            // GGUF metadata's `<arch>.context_length` and rope_scaling
            // hash often disagree with the canonical HF config.json
            // (e.g. Qwen2.5-0.5B-Instruct GGUF reports 8192, JSON
            // reports 32768; unsloth Llama-3 GGUFs omit rope_scaling
            // entirely so we infer it). Treat those config-disambiguation
            // hints as permissive (None) for GGUF sources — the
            // fingerprint's positive shape + suffix checks plus
            // `is_gguf_gate` are already disjoint enough; the
            // compile-time variant's baked rope/max_pos values stay
            // authoritative.
            let suppress_hf_hints = weights.is_gguf();
            let hf_fp = ferrite_forward::HfFingerprint {
                max_position_embeddings: if suppress_hf_hints {
                    None
                } else {
                    hf_config.max_position_embeddings.map(|v| v as u64)
                },
                rope_scaling_type: if suppress_hf_hints {
                    None
                } else {
                    hf_config
                        .extra
                        .get("rope_scaling")
                        .and_then(|rs| rs.get("rope_type").or_else(|| rs.get("type")))
                        .and_then(|v| v.as_str())
                },
                rope_scaling_hash: if suppress_hf_hints {
                    None
                } else {
                    hf_config
                        .extra
                        .get("rope_scaling")
                        .map(ferrite_forward::hash_json_value)
                },
            };
            // Runtime `max_model_len` — matches the serve-level
            // resolution (CLI `--max-model-len` ∨ HF
            // `max_position_embeddings` ∨ 4096) so Phi-3 LongRoPE's
            // `use_long_rope = max_model_len > original_max_pos`
            // flag agrees with Python vLLM at init time.
            let max_model_len = self
                .config
                .max_model_len
                .or(hf_config.max_position_embeddings)
                .unwrap_or(4096);
            // Pass the runtime (tp_world_size, tp_rank) so try_load
            // picks the matching `(arch, tp)` registration and the
            // per-(model, tp) emitted `Weights::load` body shards
            // weights by `tp_rank`. tp>1 is now end-to-end wired:
            // codegen's shard-kind dispatch (52615e881) routes load
            // calls through `_sharded` helpers; lowering injects
            // AllReduce after vocab-parallel Embed (aefa1de37) and
            // AllGather after lm_head Gemm (8a3ea25bc). At tp=1 every
            // emitted call site is byte-equivalent to before TP
            // landed — `_sharded` variants degrade to the unsharded
            // helpers when world == 1.
            let ferrite_tp = u8::try_from(tp_world).unwrap_or(1);
            let ferrite_rank = u8::try_from(tp_rank).unwrap_or(0);
            if let Some(ferrite_weights) = ferrite_forward::try_load(
                &mut weights,
                stream,
                arch.as_str(),
                ferrite_tp,
                ferrite_rank,
                max_model_len,
                hf_fp,
            )
            .map_err(|e| ExecutorError::WorkerInit(format!("ferrite-forward load: {e}")))?
            {
                // Ferrite owns rotary construction: the emitted
                // `Weights::load` built the RotaryCache (plus any
                // dual-rotary arch-local variant) inline from the
                // manifest's bounds/scalars/rope_scaling. Nothing
                // for the executor to do.
                info!(
                    "CudaWorker: loaded {} via ferrite-forward ({})",
                    arch,
                    ferrite_weights.arch_name(),
                );
                // Probe for a sibling `MultimodalForward` registration.
                // Same `(arch_hint, tp_world_size)` filter as the text
                // try_load above; returns `Ok(None)` for text-only
                // arches and for MM-capable arches whose live
                // checkpoint has no `visual.*` tensors. The handle
                // rides on `FerriteModel.mm` and is consumed at
                // forward time when the batch carries `mm_data`.
                let mm = ferrite_forward::try_load_mm(
                    &mut weights,
                    stream,
                    arch.as_str(),
                    ferrite_tp,
                    ferrite_rank,
                    max_model_len,
                    hf_fp,
                )
                .map_err(|e| ExecutorError::WorkerInit(format!("ferrite-forward MM load: {e}")))?;
                if mm.is_some() {
                    info!(
                        "CudaWorker: loaded {} vision encoder via ferrite-forward",
                        arch
                    );
                }
                // Hybrid arches need an auxiliary recurrent-state pool
                // alongside the paged KV cache. Today only `qwen3_next`
                // qualifies; the existing `gdn_state_pool` /
                // `qwen3_next_config` worker fields back both the
                // ferrite and the hand-written paths. We populate the
                // config here so `init_kv_cache_pool` builds the pool
                // for ferrite loads too. New hybrid arches would extend
                // this branch (or, longer term, a `FerriteWeights`
                // method that exposes the per-arch state requirements).
                if ferrite_weights.arch_name() == "qwen3_next" {
                    self.qwen3_next_config = Some(qwen3_next_config_from_hf(&hf_config)?);
                }
                Some(CudaModel::Ferrite(Box::new(FerriteModel {
                    weights: ferrite_weights,
                    mm,
                    #[cfg(feature = "nccl")]
                    tp_group: None,
                    tp_world_size: self.config.tp_world_size,
                })))
            } else {
                None
            }
        } else {
            None
        };
        let model = if let Some(m) = ferrite_loaded {
            m
        } else {
            // GGUF source has only one supported load path
            // (ferrite-forward); the fallback paths below read from
            // the safetensors mmap map and would fail deep with a
            // confusing "weight not found". Surface the real problem:
            // ferrite-forward did not match a compiled variant.
            if uses_ggml {
                return Err(ExecutorError::WorkerInit(format!(
                    "GGUF source loaded but no ferrite-forward variant matched arch=`{arch}` \
                     at tp={tp_world}. Make sure the `-ggml` overlay is compiled in: \
                     rebuild without `FERRITE_MODELS` (compiles every variant) or set \
                     `FERRITE_MODELS=<base-stem>` (the prefix matches `<base>-ggml` too — \
                     e.g. `FERRITE_MODELS=llama-3.2-3b`). Also confirm the arch's \
                     `configs/quantizations.json` lists `\"ggml\"`."
                )));
            }
            match arch.as_str() {
                // Qwen3 dense — Llama math + per-head Q/K rmsnorm before
                // RoPE (no QKV bias). Only bf16/fp16 ferrite-forward is
                // supported; quant / TP / PP / FERRITE_DISABLE would
                // silently drop the q_norm / k_norm tensors (the hand-
                // written `LlamaForCausalLM` doesn't handle them) — hard
                // error here until a dedicated path lands.
                "Qwen3ForCausalLM" => {
                    return Err(ExecutorError::WorkerInit(
                        "Qwen3 dense: only bf16/fp16 ferrite-forward path is supported \
                     (quant / TP / PP / FERRITE_DISABLE require the hand-written \
                     model, which doesn't yet handle Qwen3's per-head q_norm/k_norm)"
                            .into(),
                    ));
                }
                "LlamaForCausalLM" | "MistralForCausalLM" | "Phi3ForCausalLM" => {
                    let config = llama_config_from_hf(&hf_config)?;
                    // ferrite-forward already handled dense + AWQ above.
                    // This arm is the hand-written quant/TP/PP fallback.
                    {
                        let m = if qconfig.is_bnb4bit() {
                            let bnb_cfg = match &qconfig {
                                vllm_cuda::quant::QuantConfig::Bnb4bit(c) => c,
                                _ => unreachable!(),
                            };
                            vllm_cuda::model::llama::LlamaForCausalLM::load_bnb4bit(
                                &mut weights,
                                &config,
                                dtype,
                                bnb_cfg,
                                device,
                            )
                        } else if qconfig.is_fp8() {
                            let fp8_cfg = match &qconfig {
                                vllm_cuda::quant::QuantConfig::Fp8(c) => c,
                                _ => unreachable!(),
                            };
                            if fp8_cfg.weight_block_size.is_some() {
                                if use_tp {
                                    vllm_cuda::model::llama::LlamaForCausalLM::load_fp8_block_tp(
                                        &mut weights,
                                        &config,
                                        dtype,
                                        config.rms_norm_eps,
                                        tp,
                                        device,
                                    )
                                } else {
                                    vllm_cuda::model::llama::LlamaForCausalLM::load_fp8_block(
                                        &mut weights,
                                        &config,
                                        dtype,
                                        config.rms_norm_eps,
                                        device,
                                    )
                                }
                            } else if use_tp {
                                vllm_cuda::model::llama::LlamaForCausalLM::load_fp8_tp(
                                    &mut weights,
                                    &config,
                                    dtype,
                                    config.rms_norm_eps,
                                    tp,
                                    device,
                                )
                            } else {
                                vllm_cuda::model::llama::LlamaForCausalLM::load_fp8(
                                    &mut weights,
                                    &config,
                                    dtype,
                                    config.rms_norm_eps,
                                    device,
                                )
                            }
                        } else if qconfig.is_quantized() {
                            vllm_cuda::model::llama::LlamaForCausalLM::load_quantized(
                                &mut weights,
                                &config,
                                dtype,
                                &qconfig,
                                device,
                            )
                        } else if use_tp && use_pp {
                            vllm_cuda::model::llama::LlamaForCausalLM::load_tp_pp(
                                &mut weights,
                                &config,
                                dtype,
                                tp,
                                pp_config.unwrap(),
                                device,
                            )
                        } else if use_pp {
                            vllm_cuda::model::llama::LlamaForCausalLM::load_pp(
                                &mut weights,
                                &config,
                                dtype,
                                pp_config.unwrap(),
                                device,
                            )
                        } else if use_tp {
                            vllm_cuda::model::llama::LlamaForCausalLM::load_tp(
                                &mut weights,
                                &config,
                                dtype,
                                tp,
                                device,
                            )
                        } else {
                            vllm_cuda::model::llama::LlamaForCausalLM::load(
                                &mut weights,
                                &config,
                                dtype,
                                device,
                            )
                        }
                        .map_err(|e| {
                            ExecutorError::WorkerInit(format!("LlamaForCausalLM load: {e}"))
                        })?;
                        CudaModel::Llama(m)
                    } // close the else { legacy } block
                }
                "Qwen2ForCausalLM" | "Qwen2_5ForCausalLM" => {
                    let llama_config = llama_config_from_hf(&hf_config)?;
                    let qwen2_config =
                        vllm_cuda::model::qwen2::Qwen2Config::from_llama_config(llama_config);
                    // ferrite-forward already handled dense + AWQ above.
                    // This arm is the hand-written quant/TP/PP fallback.
                    {
                        let m = if qconfig.is_bnb4bit() {
                            let bnb_cfg = match &qconfig {
                                vllm_cuda::quant::QuantConfig::Bnb4bit(c) => c,
                                _ => unreachable!(),
                            };
                            vllm_cuda::model::qwen2::Qwen2ForCausalLM::load_bnb4bit(
                                &mut weights,
                                &qwen2_config,
                                dtype,
                                bnb_cfg,
                                device,
                            )
                        } else if qconfig.is_fp8() {
                            let fp8_cfg = match &qconfig {
                                vllm_cuda::quant::QuantConfig::Fp8(c) => c,
                                _ => unreachable!(),
                            };
                            if fp8_cfg.weight_block_size.is_some() {
                                if use_tp {
                                    vllm_cuda::model::qwen2::Qwen2ForCausalLM::load_fp8_block_tp(
                                        &mut weights,
                                        &qwen2_config,
                                        dtype,
                                        tp,
                                        device,
                                    )
                                } else {
                                    vllm_cuda::model::qwen2::Qwen2ForCausalLM::load_fp8(
                                        &mut weights,
                                        &qwen2_config,
                                        dtype,
                                        device,
                                    )
                                }
                            } else if use_tp {
                                vllm_cuda::model::qwen2::Qwen2ForCausalLM::load_fp8_tp(
                                    &mut weights,
                                    &qwen2_config,
                                    dtype,
                                    tp,
                                    device,
                                )
                            } else {
                                vllm_cuda::model::qwen2::Qwen2ForCausalLM::load_fp8(
                                    &mut weights,
                                    &qwen2_config,
                                    dtype,
                                    device,
                                )
                            }
                        } else if qconfig.is_quantized() {
                            vllm_cuda::model::qwen2::Qwen2ForCausalLM::load_quantized(
                                &mut weights,
                                &qwen2_config,
                                dtype,
                                &qconfig,
                                device,
                            )
                        } else if use_tp && use_pp {
                            vllm_cuda::model::qwen2::Qwen2ForCausalLM::load_tp_pp(
                                &mut weights,
                                &qwen2_config,
                                dtype,
                                tp,
                                pp_config.unwrap(),
                                device,
                            )
                        } else if use_pp {
                            vllm_cuda::model::qwen2::Qwen2ForCausalLM::load_pp(
                                &mut weights,
                                &qwen2_config,
                                dtype,
                                pp_config.unwrap(),
                                device,
                            )
                        } else if use_tp {
                            vllm_cuda::model::qwen2::Qwen2ForCausalLM::load_tp(
                                &mut weights,
                                &qwen2_config,
                                dtype,
                                tp,
                                device,
                            )
                        } else {
                            vllm_cuda::model::qwen2::Qwen2ForCausalLM::load(
                                &mut weights,
                                &qwen2_config,
                                dtype,
                                device,
                            )
                        }
                        .map_err(|e| ExecutorError::WorkerInit(format!("Qwen2 load: {e}")))?;
                        CudaModel::Qwen2(m)
                    } // close else { legacy }
                }
                "Gemma2ForCausalLM" => {
                    let config = gemma2_config_from_hf(&hf_config)?;
                    // ferrite-forward already handled dense above.
                    // This arm is the hand-written quant/TP/PP fallback.
                    {
                        let m = if qconfig.is_bnb4bit() {
                            let bnb_cfg = match &qconfig {
                                vllm_cuda::quant::QuantConfig::Bnb4bit(c) => c,
                                _ => unreachable!(),
                            };
                            vllm_cuda::model::gemma2::Gemma2ForCausalLM::load_bnb4bit(
                                &mut weights,
                                &config,
                                dtype,
                                bnb_cfg,
                                device,
                            )
                        } else if qconfig.is_fp8() {
                            if use_tp {
                                vllm_cuda::model::gemma2::Gemma2ForCausalLM::load_fp8_tp(
                                    &mut weights,
                                    &config,
                                    dtype,
                                    tp,
                                    device,
                                )
                            } else {
                                vllm_cuda::model::gemma2::Gemma2ForCausalLM::load_fp8(
                                    &mut weights,
                                    &config,
                                    dtype,
                                    device,
                                )
                            }
                        } else if qconfig.is_quantized() {
                            vllm_cuda::model::gemma2::Gemma2ForCausalLM::load_quantized(
                                &mut weights,
                                &config,
                                dtype,
                                &qconfig,
                                device,
                            )
                        } else if use_tp && use_pp {
                            vllm_cuda::model::gemma2::Gemma2ForCausalLM::load_tp_pp(
                                &mut weights,
                                &config,
                                dtype,
                                tp,
                                pp_config.unwrap(),
                                device,
                            )
                        } else if use_pp {
                            vllm_cuda::model::gemma2::Gemma2ForCausalLM::load_pp(
                                &mut weights,
                                &config,
                                dtype,
                                pp_config.unwrap(),
                                device,
                            )
                        } else if use_tp {
                            vllm_cuda::model::gemma2::Gemma2ForCausalLM::load_tp(
                                &mut weights,
                                &config,
                                dtype,
                                tp,
                                device,
                            )
                        } else {
                            vllm_cuda::model::gemma2::Gemma2ForCausalLM::load(
                                &mut weights,
                                &config,
                                dtype,
                                device,
                            )
                        }
                        .map_err(|e| ExecutorError::WorkerInit(format!("Gemma2 load: {e}")))?;
                        CudaModel::Gemma2(m)
                    } // close else { legacy }
                }
                "Gemma3ForCausalLM" | "Gemma3ForConditionalGeneration" => {
                    // For Gemma3ForConditionalGeneration (multimodal), resolve the
                    // nested text_config so we get the text backbone parameters.
                    let effective_hf = if arch == "Gemma3ForConditionalGeneration" {
                        if let Some(tc) = hf_config.extra.get("text_config") {
                            serde_json::from_value::<HfModelConfig>(tc.clone()).map_err(|e| {
                                ExecutorError::WorkerInit(format!(
                                    "failed to parse Gemma3 text_config: {e}"
                                ))
                            })?
                        } else {
                            hf_config.clone()
                        }
                    } else {
                        hf_config.clone()
                    };
                    let config = gemma3_config_from_hf(&effective_hf)?;
                    // Multimodal model: strip "language_model." prefix from weights.
                    if arch == "Gemma3ForConditionalGeneration" {
                        weights.strip_prefix("language_model.");
                    }
                    let m = if use_tp && use_pp {
                        vllm_cuda::model::gemma3::Gemma3ForCausalLM::load_tp_pp(
                            &mut weights,
                            &config,
                            dtype,
                            tp,
                            pp_config.unwrap(),
                            device,
                        )
                    } else if use_pp {
                        vllm_cuda::model::gemma3::Gemma3ForCausalLM::load_pp(
                            &mut weights,
                            &config,
                            dtype,
                            pp_config.unwrap(),
                            device,
                        )
                    } else if use_tp {
                        vllm_cuda::model::gemma3::Gemma3ForCausalLM::load_tp(
                            &mut weights,
                            &config,
                            dtype,
                            tp,
                            device,
                        )
                    } else {
                        vllm_cuda::model::gemma3::Gemma3ForCausalLM::load(
                            &mut weights,
                            &config,
                            dtype,
                            device,
                        )
                    }
                    .map_err(|e| ExecutorError::WorkerInit(format!("Gemma3 load: {e}")))?;
                    CudaModel::Gemma3(m)
                }
                "GraniteForCausalLM" => {
                    let config = llama_config_from_hf(&hf_config)?;
                    // ferrite-forward already handled dense above. This
                    // arm is the hand-written quant/TP/PP fallback;
                    // Granite's scalar multipliers (embedding/residual/
                    // logits) are applied post-load to the inner Llama
                    // weights.
                    {
                        let mut m = if qconfig.is_bnb4bit() {
                            let bnb_cfg = match &qconfig {
                                vllm_cuda::quant::QuantConfig::Bnb4bit(c) => c,
                                _ => unreachable!(),
                            };
                            vllm_cuda::model::llama::LlamaForCausalLM::load_bnb4bit(
                                &mut weights,
                                &config,
                                dtype,
                                bnb_cfg,
                                device,
                            )
                        } else if qconfig.is_fp8() {
                            let fp8_cfg = match &qconfig {
                                vllm_cuda::quant::QuantConfig::Fp8(c) => c,
                                _ => unreachable!(),
                            };
                            if fp8_cfg.weight_block_size.is_some() {
                                if use_tp {
                                    vllm_cuda::model::llama::LlamaForCausalLM::load_fp8_block_tp(
                                        &mut weights,
                                        &config,
                                        dtype,
                                        config.rms_norm_eps,
                                        tp,
                                        device,
                                    )
                                } else {
                                    vllm_cuda::model::llama::LlamaForCausalLM::load_fp8_block(
                                        &mut weights,
                                        &config,
                                        dtype,
                                        config.rms_norm_eps,
                                        device,
                                    )
                                }
                            } else if use_tp {
                                vllm_cuda::model::llama::LlamaForCausalLM::load_fp8_tp(
                                    &mut weights,
                                    &config,
                                    dtype,
                                    config.rms_norm_eps,
                                    tp,
                                    device,
                                )
                            } else {
                                vllm_cuda::model::llama::LlamaForCausalLM::load_fp8(
                                    &mut weights,
                                    &config,
                                    dtype,
                                    config.rms_norm_eps,
                                    device,
                                )
                            }
                        } else if qconfig.is_quantized() {
                            vllm_cuda::model::llama::LlamaForCausalLM::load_quantized(
                                &mut weights,
                                &config,
                                dtype,
                                &qconfig,
                                device,
                            )
                        } else if use_tp {
                            vllm_cuda::model::llama::LlamaForCausalLM::load_tp(
                                &mut weights,
                                &config,
                                dtype,
                                tp,
                                device,
                            )
                        } else {
                            vllm_cuda::model::llama::LlamaForCausalLM::load(
                                &mut weights,
                                &config,
                                dtype,
                                device,
                            )
                        }
                        .map_err(|e| {
                            ExecutorError::WorkerInit(format!("GraniteForCausalLM load: {e}"))
                        })?;

                        // Parse Granite-specific multipliers from config.json extras.
                        let extra = &hf_config.extra;
                        let embedding_multiplier = extra
                            .get("embedding_multiplier")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(1.0)
                            as f32;
                        let residual_multiplier = extra
                            .get("residual_multiplier")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(1.0)
                            as f32;
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
                    } // close else { legacy }
                }
                "MixtralForCausalLM" => {
                    let config = mixtral_config_from_hf(&hf_config)?;
                    let m = if qconfig.is_quantized() && !qconfig.is_fp8() && !qconfig.is_bnb4bit()
                    {
                        vllm_cuda::model::mixtral::MixtralForCausalLM::load_quantized(
                            &mut weights,
                            &config,
                            dtype,
                            &qconfig,
                            device,
                        )
                    } else if qconfig.is_fp8() {
                        vllm_cuda::model::mixtral::MixtralForCausalLM::load_fp8(
                            &mut weights,
                            &config,
                            dtype,
                            device,
                        )
                    } else if use_tp {
                        vllm_cuda::model::mixtral::MixtralForCausalLM::load_tp(
                            &mut weights,
                            &config,
                            dtype,
                            tp,
                            device,
                        )
                    } else {
                        vllm_cuda::model::mixtral::MixtralForCausalLM::load(
                            &mut weights,
                            &config,
                            dtype,
                            device,
                        )
                    }
                    .map_err(|e| ExecutorError::WorkerInit(format!("Mixtral load: {e}")))?;
                    CudaModel::Mixtral(m)
                }
                "Qwen2MoeForCausalLM" => {
                    let config = qwen2_moe_config_from_hf(&hf_config)?;
                    let m = if qconfig.is_quantized() && !qconfig.is_fp8() && !qconfig.is_bnb4bit()
                    {
                        vllm_cuda::model::qwen2_moe::Qwen2MoeForCausalLM::load_quantized(
                            &mut weights,
                            &config,
                            dtype,
                            &qconfig,
                            device,
                        )
                    } else if qconfig.is_fp8() {
                        vllm_cuda::model::qwen2_moe::Qwen2MoeForCausalLM::load_fp8(
                            &mut weights,
                            &config,
                            dtype,
                            device,
                        )
                    } else if use_tp {
                        vllm_cuda::model::qwen2_moe::Qwen2MoeForCausalLM::load_tp(
                            &mut weights,
                            &config,
                            dtype,
                            tp,
                            device,
                        )
                    } else {
                        vllm_cuda::model::qwen2_moe::Qwen2MoeForCausalLM::load(
                            &mut weights,
                            &config,
                            dtype,
                            device,
                        )
                    }
                    .map_err(|e| ExecutorError::WorkerInit(format!("Qwen2MoE load: {e}")))?;
                    CudaModel::Qwen2Moe(m)
                }
                "Qwen3MoeForCausalLM" => {
                    let config = qwen2_moe_config_from_hf(&hf_config)?;
                    let m = if qconfig.is_quantized() && !qconfig.is_fp8() && !qconfig.is_bnb4bit()
                    {
                        vllm_cuda::model::qwen3_moe::Qwen3MoeForCausalLM::load_quantized(
                            &mut weights,
                            &config,
                            dtype,
                            &qconfig,
                            device,
                        )
                    } else if qconfig.is_fp8() {
                        let fp8_cfg = match &qconfig {
                            vllm_cuda::quant::QuantConfig::Fp8(c) => c,
                            _ => unreachable!(),
                        };
                        if fp8_cfg.weight_block_size.is_some() {
                            let tp_opt = if use_tp { Some(tp) } else { None };
                            vllm_cuda::model::qwen3_moe::Qwen3MoeForCausalLM::load_fp8_block(
                                &mut weights,
                                &config,
                                dtype,
                                tp_opt,
                                device,
                            )
                        } else {
                            vllm_cuda::model::qwen3_moe::Qwen3MoeForCausalLM::load_fp8(
                                &mut weights,
                                &config,
                                dtype,
                                device,
                            )
                        }
                    } else if use_tp {
                        vllm_cuda::model::qwen3_moe::Qwen3MoeForCausalLM::load_tp(
                            &mut weights,
                            &config,
                            dtype,
                            tp,
                            device,
                        )
                    } else {
                        vllm_cuda::model::qwen3_moe::Qwen3MoeForCausalLM::load(
                            &mut weights,
                            &config,
                            dtype,
                            device,
                        )
                    }
                    .map_err(|e| ExecutorError::WorkerInit(format!("Qwen3MoE load: {e}")))?;
                    CudaModel::Qwen3Moe(m)
                }
                "Qwen3NextForCausalLM" => {
                    let config = qwen3_next_config_from_hf(&hf_config)?;
                    let m = vllm_cuda::model::qwen3_next::Qwen3NextForCausalLM::load(
                        &mut weights,
                        &config,
                        dtype,
                        device,
                    )
                    .map_err(|e| ExecutorError::WorkerInit(format!("Qwen3Next load: {e}")))?;
                    self.qwen3_next_config = Some(config);
                    CudaModel::Qwen3Next(m)
                }
                "DeepseekV2ForCausalLM" | "DeepSeekV3ForCausalLM" | "DeepseekV3ForCausalLM" => {
                    let config = deepseek_v2_config_from_hf(&hf_config)?;
                    let m = if use_tp {
                        vllm_cuda::model::deepseek_v2::DeepSeekV2ForCausalLM::load_tp(
                            &mut weights,
                            &config,
                            dtype,
                            tp,
                            device,
                        )
                    } else {
                        vllm_cuda::model::deepseek_v2::DeepSeekV2ForCausalLM::load(
                            &mut weights,
                            &config,
                            dtype,
                            device,
                        )
                    }
                    .map_err(|e| ExecutorError::WorkerInit(format!("DeepSeekV2 load: {e}")))?;
                    CudaModel::DeepSeekV2(m)
                }
                "CohereForCausalLM" => {
                    let config = commandr_config_from_hf(&hf_config)?;
                    let m = if qconfig.is_bnb4bit() {
                        let bnb_cfg = match &qconfig {
                            vllm_cuda::quant::QuantConfig::Bnb4bit(c) => c,
                            _ => unreachable!(),
                        };
                        vllm_cuda::model::commandr::CommandRForCausalLM::load_bnb4bit(
                            &mut weights,
                            &config,
                            dtype,
                            bnb_cfg,
                            device,
                        )
                    } else if qconfig.is_fp8() {
                        vllm_cuda::model::commandr::CommandRForCausalLM::load_fp8(
                            &mut weights,
                            &config,
                            dtype,
                            device,
                        )
                    } else {
                        vllm_cuda::model::commandr::CommandRForCausalLM::load(
                            &mut weights,
                            &config,
                            dtype,
                            device,
                        )
                    }
                    .map_err(|e| ExecutorError::WorkerInit(format!("CommandR load: {e}")))?;
                    CudaModel::CommandR(m)
                }
                _ => {
                    return Err(ExecutorError::WorkerInit(format!(
                        "unsupported architecture for cuda-backend: {arch}. \
                     Supported: LlamaForCausalLM, MistralForCausalLM, Qwen3ForCausalLM, \
                     Phi3ForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM, Gemma3ForCausalLM, \
                     Gemma3ForConditionalGeneration, GraniteForCausalLM, MixtralForCausalLM, \
                     Qwen2MoeForCausalLM, Qwen3MoeForCausalLM, CohereForCausalLM, \
                     Qwen3NextForCausalLM, DeepseekV2ForCausalLM, DeepSeekV3ForCausalLM, \
                     DeepseekV3ForCausalLM, ModernBertModel, ModernBertForMaskedLM \
                     (last two via ferrite-forward)"
                    )));
                }
            }
        };

        // Sync to ensure all async H2D weight copies are complete.
        unsafe { driver::stream_synchronize(device.compute_stream) }
            .map_err(|e| ExecutorError::WorkerInit(format!("weight sync: {e}")))?;

        // Collect GPU weight allocation pointers for sleep/wake lifecycle.
        self.weight_gpu_allocs = weights.take_gpu_allocs();
        drop(weights); // CPU mmaps freed, GPU memory owned by model layers

        self.model_dtype = dtype;
        self.resolved_architecture = Some(arch);
        self.model = Some(model);
        self.uses_ggml = uses_ggml;
        self.model_dir = Some(model_dir.clone());
        self.hf_config = Some(hf_config);
        self.pp_config = pp_config;

        // Resolve pooling strategy.
        self.pooling_strategy = match self.config.pooling_strategy.as_str() {
            "last" => vllm_model::embedding::PoolingStrategy::Last,
            "cls" => vllm_model::embedding::PoolingStrategy::Cls,
            "mean" => vllm_model::embedding::PoolingStrategy::Mean,
            "all" | "all_tokens" => vllm_model::embedding::PoolingStrategy::AllTokens,
            _ => {
                // "auto": detect from 1_Pooling/config.json, default to Last.
                vllm_model::embedding::detect_pooling_strategy(&model_dir)
                    .unwrap_or(vllm_model::embedding::PoolingStrategy::Last)
            }
        };

        // Collect tokenizer.
        if let Ok(Some(tok)) = tokenizer_handle.join() {
            self.preloaded_tokenizer = Some(tok);
        }

        // Initialize logits processor pipeline (grammar handled separately).
        let vocab_size = self.model.as_ref().unwrap().vocab_size();
        let processors: Vec<Box<dyn vllm_cuda::logits_processor::LogitsProcessor>> = vec![
            Box::new(MinTokensProcessor::new()),
            Box::new(LogitBiasProcessor::new()),
            Box::new(PenaltiesProcessor::new(vocab_size)),
            Box::new(BadWordsProcessor::new()),
        ];
        self.logits_pipeline = Some(LogitsProcessorPipeline::new(processors));

        // Reinitialize SealPadProcessor with real EOS token IDs from model config.
        if let Some(ref hf_config) = self.hf_config {
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
            self.seal_pad_processor =
                SealPadProcessor::new(eos_token_ids, 0, self.config.block_size);
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

        // Resolve KV cache dtype: FP8 E4M3 when configured, otherwise model dtype.
        let kv_dtype = if self.kv_cache_is_fp8 {
            GpuDType::Fp8E4m3
        } else {
            self.model_dtype
        };

        let pool = unsafe {
            KvCachePool::new(
                model.num_layers(),
                num_gpu_blocks,
                self.config.block_size,
                model.num_kv_heads(),
                model.head_dim(),
                kv_dtype,
            )
        }
        .map_err(|e| ExecutorError::WorkerInit(format!("KvCachePool: {e}")))?;

        self.kv_cache = Some(pool);

        // Allocate GDN state pool for Qwen3Next.
        if let Some(ref config) = self.qwen3_next_config {
            let dev = self.device.as_ref().unwrap();
            let gdn_pool = unsafe {
                vllm_cuda::model::qwen3_next::make_gdn_state_pool(
                    config,
                    num_gpu_blocks,
                    dev.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerInit(format!("GdnStatePool: {e}")))?;
            self.gdn_state_pool = Some(gdn_pool);
        }

        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        if let Some(ref dev) = self.device {
            unsafe { driver::ctx_set_current(dev.ctx) }
                .map_err(|e| ExecutorError::WorkerInit(format!("ctx_set_current: {e}")))?;
        }

        // Skip profiling forward for models where the dummy forward is
        // incompatible or too expensive:
        // - GGML: flash attention triggers illegal memory access
        // - Qwen3Next: similar profiling issues
        // - PP: multi-rank profiling not supported
        // - MoE: fused path uses ~55 MB/layer (OK), but profiling with
        //   max_num_batched_tokens can still OOM on the attention side
        let pp_active = self.pp_config.is_some_and(|pp| pp.pp_size > 1);
        let is_moe = self.model.as_ref().is_some_and(|m| m.is_moe());
        let is_encoder = self.model.as_ref().is_some_and(|m| m.is_ferrite_encoder());
        if self.uses_ggml || self.qwen3_next_config.is_some() || pp_active || is_moe || is_encoder {
            let tag = if self.uses_ggml {
                "GGML"
            } else if pp_active {
                "PP"
            } else if is_moe {
                "MoE"
            } else if is_encoder {
                "encoder"
            } else {
                "Qwen3Next"
            };
            info!("CudaWorker: {tag} model — skipping activation profiling, using fixed estimate");
            let (free, total) = cudarc::driver::result::mem_get_info()
                .map_err(|e| ExecutorError::WorkerInit(format!("cuMemGetInfo: {e}")))?;
            let weights_and_overhead = total.saturating_sub(free);
            let peak_activation_estimate = 512 * 1024 * 1024; // 512 MB conservative
            let utilization = self.config.gpu_memory_utilization;
            let available = compute_available_kv_bytes(
                total,
                weights_and_overhead,
                peak_activation_estimate,
                utilization,
            );
            info!(
                "Memory estimate: total={:.1} GiB, weights+overhead={:.1} GiB, \
                 est_activations=512 MiB",
                total as f64 / 1_073_741_824.0,
                weights_and_overhead as f64 / 1_073_741_824.0,
            );
            return Ok(available);
        }

        // Like Python vLLM: profile peak activation memory with a dummy forward
        // pass, then subtract it from available memory for KV cache sizing.
        //
        // Python uses `allocated_bytes.all.peak` (peak ACTIVE PyTorch allocations
        // during the profiling forward) + `non_torch_increase` (non-PyTorch CUDA
        // memory that persisted after the forward, e.g. cuBLAS workspace).
        //
        // We mirror this with:
        //   torch_peak     = caching.peak_active_bytes()  (peak live allocations)
        //   non_torch      = (total - free_after_trim) - caching.memory_reserved()
        //   peak_activations = torch_peak + non_torch
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

        // Allocate dummy inputs for the profiling forward pass.
        //
        // IDs and positions MUST be zero-initialized — the caching allocator
        // recycles freed blocks whose contents are undefined.  Garbage token
        // IDs cause out-of-bounds embedding lookups; garbage positions cause
        // out-of-bounds RoPE lookups.  Both segfault.  (Matches Python vLLM
        // which uses torch.zeros for its profiling dummy tensors.)
        let dummy_ids = device.alloc_gpu_tensor_zeroed(&[prefill_tokens], GpuDType::U32);
        let dummy_pos = device.alloc_gpu_tensor_zeroed(&[prefill_tokens], GpuDType::U32);

        // Slot mapping: slot[i] = i (sequential)
        let slot_data: Vec<i64> = (0..prefill_tokens as i64).collect();
        let dummy_slots =
            device.alloc_gpu_tensor_from_host(&[prefill_tokens], GpuDType::I64, unsafe {
                std::slice::from_raw_parts(slot_data.as_ptr() as *const u8, prefill_tokens * 8)
            });

        let cu_q: Vec<u32> = vec![0, prefill_tokens as u32];
        let gpu_cu_q = device.alloc_gpu_tensor_from_host(&[2], GpuDType::U32, unsafe {
            std::slice::from_raw_parts(cu_q.as_ptr() as *const u8, cu_q.len() * 4)
        });

        let seqused_data: Vec<u32> = vec![prefill_tokens as u32];
        let dummy_seqused = device.alloc_gpu_tensor_from_host(&[1], GpuDType::U32, unsafe {
            std::slice::from_raw_parts(seqused_data.as_ptr() as *const u8, 4)
        });

        // Block table: [1, num_blocks_needed] — sequential block indices
        let num_blocks_needed = prefill_tokens.div_ceil(self.config.block_size);
        let bt_data: Vec<u32> = (0..num_blocks_needed as u32).collect();
        let dummy_bt =
            device.alloc_gpu_tensor_from_host(&[1, num_blocks_needed], GpuDType::U32, unsafe {
                std::slice::from_raw_parts(bt_data.as_ptr() as *const u8, num_blocks_needed * 4)
            });

        // Create a KV cache large enough for the profiling tokens.
        let dummy_kv = unsafe {
            vllm_cuda::KvCachePool::new(
                model.num_layers(),
                num_blocks_needed,
                self.config.block_size,
                model.num_kv_heads(),
                model.head_dim(),
                self.model_dtype,
            )
        }
        .map_err(|e| ExecutorError::WorkerInit(format!("dummy KvCachePool: {e}")))?;

        // Reset peak stats immediately before the profiling forward so we measure
        // only the allocations made during this forward pass.
        device.caching.reset_peak_stats();

        // Run the forward pass to warm up cuBLAS and measure peak memory.
        unsafe {
            let _ = model.forward(
                TensorView::from_raw(dummy_ids),
                TensorView::from_raw(dummy_pos),
                TensorView::from_raw(dummy_slots),
                TensorView::from_raw(gpu_cu_q),
                TensorView::from_raw(dummy_seqused),
                TensorView::from_raw(dummy_bt),
                prefill_tokens,
                prefill_tokens,
                &dummy_kv,
                device,
                None,
                None,
            );
            if let Err(e) = driver::stream_synchronize(device.compute_stream) {
                tracing::error!("Memory profiling forward failed: {e}");
            }
        }

        // Capture peak active bytes from caching allocator — mirrors Python's
        // `allocated_bytes.all.peak` (peak ACTIVE allocations, freed blocks not
        // counted).
        let torch_peak = device.caching.peak_active_bytes();

        // Free the dummy KV cache and all cached allocator blocks, then trim to
        // return segments to the driver so cuMemGetInfo reflects only permanent
        // allocations (cuBLAS workspace, NCCL, etc.).
        drop(dummy_kv);
        device.caching.trim();

        // non_torch_increase = memory permanently held outside the caching
        // allocator after the profiling forward (cuBLAS workspace, etc.).
        // After trim(), caching.memory_reserved() == 0 for segments that were
        // released.  We compare against free_before to catch anything that
        // persisted.
        let (free_after_trim, total_memory) = cudarc::driver::result::mem_get_info()
            .map_err(|e| ExecutorError::WorkerInit(format!("cuMemGetInfo: {e}")))?;
        // memory_reserved() = bytes still held by caching allocator after trim
        // (private pool, graph capture pool, etc.)
        let caching_reserved = device.caching.memory_reserved();
        // non_torch = new permanent non-caching-allocator memory since free_before
        // (e.g. cuBLAS workspace created during forward).  Can be negative if
        // something was freed; clamp to 0.
        let used_after_trim = total_memory.saturating_sub(free_after_trim);
        let used_before = total_memory.saturating_sub(free_before);
        let non_torch = used_after_trim
            .saturating_sub(used_before)
            .saturating_sub(caching_reserved);

        let peak_activation_bytes = torch_peak + non_torch;

        // Match Python vLLM's memory calculation exactly.
        //
        // Python's flow (gpu_worker.py + mem_utils.py):
        //   non_kv_cache = weights_memory + torch_peak + non_torch + 150 MiB
        //   available_kv_bytes = requested - non_kv_cache
        //
        // Our free_before is measured AFTER model load (before profile run).
        // So (total - free_before) = weights + persistent pre-existing overhead.
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
             torch_peak={:.1} GiB, non_torch={:.1} GiB, peak_activations={:.1} GiB",
            total_memory as f64 / 1_073_741_824.0,
            weights_and_overhead as f64 / 1_073_741_824.0,
            torch_peak as f64 / 1_073_741_824.0,
            non_torch as f64 / 1_073_741_824.0,
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

        if self.uses_ggml {
            info!(
                "CudaWorker: GGML model — skipping CUDA graph capture (incompatible with graph capture)"
            );
            return Ok(());
        }

        if self.qwen3_next_config.is_some() {
            info!("CudaWorker: Qwen3Next — skipping CUDA graph capture (GDN recurrent state)");
            return Ok(());
        }

        if self.kv_cache_is_fp8 {
            info!(
                "CudaWorker: FP8 KV cache — skipping CUDA graph capture (variable scratch buffer sizes)"
            );
            return Ok(());
        }

        let max_blocks_per_seq = self.max_blocks_per_seq();

        let (mut model, mut kv_cache, mut device) =
            match (&self.model, &self.kv_cache, &mut self.device) {
                (Some(m), Some(kv), Some(d)) => (m, kv, d),
                _ => return Ok(()), // Not fully initialized yet.
            };

        // Resolve Auto mode now that we know the SM version and TP config.
        {
            let resolved = self
                .config
                .cuda_graph_mode
                .resolve(device.sm_version, self.config.tp_world_size);
            info!(
                "CudaWorker: resolved cuda_graph_mode {:?} → {:?} (SM{}, TP={})",
                self.config.cuda_graph_mode, resolved, device.sm_version, self.config.tp_world_size
            );
            self.config.cuda_graph_mode = resolved;
        }

        // Encoder models don't support CUDA graph capture (no decode loop).
        if model.is_ferrite_encoder() {
            self.config.cuda_graph_mode = CudaGraphMode::None;
            info!("CudaWorker: encoder model — disabling CUDA graphs");
        }

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

        // -----------------------------------------------------------------------
        // 1. Capture Piecewise CUDA Graphs FIRST (matching Python vLLM order)
        // -----------------------------------------------------------------------
        // Piecewise capture runs individual pieces (embedding, norm, MLP, etc.)
        // which populates cuBLAS plans and establishes memory patterns. This is
        // the safe fallback — if monolithic capture fails later (e.g. illegal
        // address from attention), piecewise graphs are already captured and the
        // CUDA context is still healthy.
        let should_capture_piecewise = matches!(
            self.config.cuda_graph_mode,
            CudaGraphMode::Piecewise | CudaGraphMode::FullAndPiecewise
        );

        if should_capture_piecewise && !capture_sizes.is_empty() {
            let num_layers = model.num_layers();
            let hidden_size = model.hidden_size();
            let piecewise_max_bs = *capture_sizes.iter().max().unwrap();
            let piecewise_capture_sizes = capture_sizes.clone();
            let piecewise_vocab_size = vocab_size;
            let piecewise_dtype = self.model_dtype;
            // Release borrowed refs from self so capture_piecewise_graphs can take &mut self.
            let _ = model;
            let _ = kv_cache;
            let _ = device;

            self.capture_piecewise_graphs(
                num_layers,
                hidden_size,
                piecewise_vocab_size,
                piecewise_dtype,
                piecewise_max_bs,
                &piecewise_capture_sizes,
            );

            // Re-obtain borrows for monolithic capture below.
            model = self.model.as_ref().unwrap();
            kv_cache = self.kv_cache.as_ref().unwrap();
            device = self.device.as_mut().unwrap();
        }

        // -----------------------------------------------------------------------
        // 2. Capture Monolithic (Full) Decode CUDA Graphs
        // -----------------------------------------------------------------------
        // Monolithic graphs capture the entire forward pass (including attention)
        // in a single CUDA graph — zero kernel launch overhead.
        //
        // MoE models are excluded: their per-layer kernel count (router +
        // N expert GEMMs + shared experts + gating) exceeds the CUDA driver's
        // undocumented graph node limit, causing CUDA_ERROR_ILLEGAL_ADDRESS that
        // permanently poisons the CUDA context. This matches Python vLLM's
        // approach of validating before capture rather than recovering after
        // failure. Piecewise graphs handle MoE models with ~1-3% decode overhead.
        let padded_max_seqlen_k: usize = 2048;
        let mut monolithic_failed = false;

        let model_supports_monolithic = !model.is_moe();
        let should_capture_monolithic = model_supports_monolithic
            && matches!(
                self.config.cuda_graph_mode,
                CudaGraphMode::Full
                    | CudaGraphMode::FullAndPiecewise
                    | CudaGraphMode::FullDecodeOnly
            );

        if !model_supports_monolithic {
            info!(
                "MoE model detected — skipping monolithic CUDA graph capture \
                 (too many graph nodes for driver limit). Piecewise graphs will be used."
            );
        }

        if !should_capture_monolithic {
            // Skip to prefill graphs / cublas autotune.
        } else {
            let mut runner = unsafe {
                CudaGraphRunner::new(max_bs, vocab_size, self.model_dtype, max_blocks_per_seq)
            }
            .map_err(|e| ExecutorError::WorkerInit(format!("CudaGraphRunner::new: {e}")))?;

            // FP8 KV cache: allocate persistent dequant buffers and cache scales.
            if self.kv_cache_is_fp8 {
                unsafe {
                    runner
                        .init_fp8_buffers(
                            padded_max_seqlen_k,
                            kv_cache.num_kv_heads,
                            kv_cache.head_dim,
                            self.model_dtype,
                        )
                        .map_err(|e| ExecutorError::WorkerInit(format!("init_fp8_buffers: {e}")))?;
                    runner
                        .cache_fp8_scales(kv_cache, device.compute_stream)
                        .map_err(|e| ExecutorError::WorkerInit(format!("cache_fp8_scales: {e}")))?;
                }
            }

            // Begin private pool for ALL graph captures (like PyTorch's shared graph pool).
            // One pool is shared across all batch sizes so blocks are reused.
            device.caching.begin_allocate_to_pool();

            // Set FP8 graph context thread-local so attention helpers use pre-allocated path.
            if let Some(ctx) = runner.fp8_graph_ctx() {
                vllm_cuda::model::attention_helpers::set_fp8_graph_ctx(ctx);
            }

            // Capture largest batch sizes first (matching Python vLLM). The first
            // capture establishes the pool's high-water mark; subsequent smaller
            // captures reuse the same memory — preventing incremental pool growth
            // that could OOM the driver.
            for &bs in capture_sizes.iter().rev() {
                info!("Capturing CUDA graph for batch_size={bs}...");
                let kv_ref = kv_cache;
                let model_ref = model;

                let result = unsafe {
                    runner.capture(bs, device, |inputs, dev| {
                        model_ref.forward(
                            TensorView::from_raw(inputs.input_ids),
                            TensorView::from_raw(inputs.positions),
                            TensorView::from_raw(inputs.slot_mapping),
                            TensorView::from_raw(inputs.cu_seqlens_q),
                            TensorView::from_raw(inputs.seqused_k),
                            TensorView::from_raw(inputs.block_table),
                            1, // max_seqlen_q = 1 for decode
                            padded_max_seqlen_k,
                            kv_ref,
                            dev,
                            None, // no last_token_indices (decode: all tokens are last)
                            None, // text-only decode: no MM splice
                        )
                    })
                };

                match result {
                    Ok(()) => info!("CUDA graph captured for batch_size={bs}"),
                    Err(e) => {
                        tracing::warn!("Failed to capture CUDA graph for bs={bs}: {e}");
                        // Stop immediately — CUDA_ERROR_ILLEGAL_ADDRESS poisons the
                        // entire CUDA context. Piecewise graphs are the fallback.
                        monolithic_failed = true;
                        break;
                    }
                }
            }

            // Clear FP8 graph context thread-local.
            vllm_cuda::model::attention_helpers::clear_fp8_graph_ctx();

            // End private pool after all captures. Blocks in the pool that are
            // still free are effectively owned by the captured graphs.
            device.caching.end_allocate_to_pool();

            if monolithic_failed {
                // Context is poisoned — discard any partially captured graphs.
                // Don't attempt staging allocation. Piecewise is the fallback.
                tracing::warn!(
                    "Discarding monolithic graphs (context poisoned). \
                 Piecewise graphs will handle all decode batches."
                );
            } else if !runner.captured_sizes().is_empty() {
                info!(
                    "CUDA graphs captured for batch sizes: {:?}",
                    runner.captured_sizes()
                );
                // Allocate pinned host staging buffers sized for the largest captured graph.
                let staging_max_bs = *runner.captured_sizes().last().unwrap();
                match unsafe { HostStaging::new(staging_max_bs, max_blocks_per_seq) } {
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
        } // end should_capture_monolithic

        // -----------------------------------------------------------------------
        // 3. Capture Prefill CUDA Graphs
        // -----------------------------------------------------------------------
        // Skip if monolithic capture poisoned the CUDA context — prefill graphs
        // use the same full forward pass and would also fail.
        if monolithic_failed {
            info!(
                "Skipping prefill graph capture (CUDA context poisoned by monolithic failure). \
                 Piecewise graphs available for decode; prefill runs eagerly."
            );
        }
        // Encoder models don't use CUDA graphs at all — skip prefill capture.
        let skip_prefill_graphs =
            monolithic_failed || self.config.cuda_graph_mode == CudaGraphMode::None;
        if skip_prefill_graphs && !monolithic_failed {
            info!("Skipping prefill graph capture (CUDA graphs disabled).");
        }
        let max_prefill_tokens = self.config.max_num_batched_tokens;
        let prefill_sizes: Vec<usize> = [128, 256, 512, 1024, 2048, 4096, 8192]
            .iter()
            .copied()
            .filter(|&s| s <= max_prefill_tokens)
            .collect();

        if !prefill_sizes.is_empty() && !skip_prefill_graphs {
            let max_prefill = *prefill_sizes.last().unwrap();
            match unsafe {
                PrefillGraphRunner::new(
                    max_prefill,
                    vocab_size,
                    self.model_dtype,
                    max_blocks_per_seq,
                )
            } {
                Ok(mut prefill_runner) => {
                    // Capture largest first (matching Python vLLM).
                    for &num_tokens in prefill_sizes.iter().rev() {
                        info!("Capturing prefill CUDA graph for num_tokens={num_tokens}...");
                        let kv_ref = kv_cache;
                        let model_ref = model;

                        let result = unsafe {
                            prefill_runner.capture(num_tokens, device, |inputs, dev| {
                                model_ref.forward(
                                    TensorView::from_raw(inputs.input_ids),
                                    TensorView::from_raw(inputs.positions),
                                    TensorView::from_raw(inputs.slot_mapping),
                                    TensorView::from_raw(inputs.cu_seqlens_q),
                                    TensorView::from_raw(inputs.seqused_k),
                                    TensorView::from_raw(inputs.block_table),
                                    num_tokens, // max_seqlen_q
                                    num_tokens, // max_seqlen_k
                                    kv_ref,
                                    dev,
                                    Some(TensorView::from_raw(inputs.last_token_indices)),
                                    None, // graph-captured prefill is text-only
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
                                break;
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

    fn sleep(&mut self, level: u32) -> ExecutorResult<()> {
        if level == 0 {
            // Level 0 = pause only, handled by the scheduler.
            return Ok(());
        }

        info!("CudaWorker: sleeping (level {level}) — freeing GPU memory");

        // Flush any deferred D2H commit.
        if self.pending_commit.take().is_some()
            && let Some(ref dev) = self.device
        {
            let _ = dev.sync_d2h();
        }

        // Save num_gpu_blocks for re-init on wake.
        if let Some(ref kv) = self.kv_cache {
            self.num_gpu_blocks_saved = kv.num_blocks;
        }

        // Drop CUDA graphs (Drop impl frees GPU memory).
        self.graph_runner = None;
        self.prefill_graph_runner = None;
        self.last_graph_batch_size = None;
        self.graph_metadata_valid = false;

        // Drop host staging (Drop impl frees pinned memory).
        self.host_staging = None;

        // Drop KV cache (Drop impl frees GPU memory).
        self.kv_cache = None;

        // Drop GDN state pool.
        self.gdn_state_pool = None;

        // Drop logits pipeline (frees GPU state).
        self.logits_pipeline = None;

        // Free weight GPU memory and drop model.
        if let Some(ref dev) = self.device {
            // Sync to ensure no in-flight ops reference weight memory.
            let _ = unsafe { driver::stream_synchronize(dev.compute_stream) };
        }
        // Drop tracked weight allocations (RawGpuMem::Drop frees GPU memory).
        self.weight_gpu_allocs.clear();
        self.model = None;

        // Clear per-request state.
        self.token_buffers.clear();
        self.prompt_lengths.clear();
        self.annotation_buffers.clear();
        self.sampling_params_map.clear();
        self.input_batch = InputBatch::new();
        self.batch_changed = false;
        self.batch_req_ids.clear();
        self.seeded_rngs.clear();
        #[cfg(feature = "guided-decoding")]
        self.grammar_states.clear();

        // Release all GPU memory held by the caching allocator (segments,
        // blocks, private pools). This returns memory to the CUDA driver and
        // resets the allocator so begin_allocate_to_pool() works on wake.
        if let Some(ref mut dev) = self.device {
            unsafe { dev.caching.release_all() };
        }

        info!("CudaWorker: sleep complete — GPU memory released");
        Ok(())
    }

    fn wake_up(&mut self, _tags: Option<&[String]>) -> ExecutorResult<()> {
        info!("CudaWorker: waking up — reloading model and KV cache");

        // Re-bind cuBLAS workspace — sleep's release_all() freed the old one.
        if let Some(ref mut dev) = self.device {
            unsafe { dev.cublas.rebind_workspace(&mut dev.caching) };
        }

        // Re-load model from disk (mmap'd safetensors, fast).
        self.load_model()?;

        // Re-init KV cache with saved block count.
        let num_gpu_blocks = self.num_gpu_blocks_saved;
        if num_gpu_blocks > 0 {
            self.initialize_cache(num_gpu_blocks, 0)?;
        }

        // Re-capture CUDA graphs.
        self.compile_or_warm_up_model()?;

        info!("CudaWorker: wake complete");
        Ok(())
    }

    fn shutdown(&mut self) {
        // Flush any deferred D2H commit before dropping the device.
        if self.pending_commit.take().is_some()
            && let Some(ref dev) = self.device
        {
            let _ = dev.sync_d2h();
        }
        // Drop tracked weight allocations (RawGpuMem::Drop frees GPU memory).
        self.weight_gpu_allocs.clear();
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
            let mut gpu_bt = Self::h2d_i32(&block_table, device)?;
            unsafe { gpu_bt.reshape(&[1, max_blocks], GpuDType::I32) };

            // Forward pass (backbone only).
            let hidden_states = unsafe {
                model.hidden_states(
                    gpu_input_ids.view(),
                    gpu_positions.view(),
                    gpu_slot_mapping.view(),
                    gpu_cu_q.view(),
                    gpu_seqused_k.view(),
                    gpu_bt.view(),
                    num_tokens,
                    num_tokens,
                    kv_cache,
                    device,
                    None,
                )
            };

            // Pool + normalize. Dereference OwnedTensor → GpuTensor (Copy).
            let embedding = Self::pool_and_normalize(*hidden_states, num_tokens, strategy, device)?;
            // Side-channel embed always uses single-vector pooling (Last/Cls/Mean).
            match embedding {
                EmbeddingData::Single(v) => results.push(v),
                EmbeddingData::Multi(_) => {
                    return Err(ExecutorError::WorkerExecution(
                        "AllTokens pooling requires --runner pooling mode".into(),
                    ));
                }
            }
            // OwnedTensor dropped here — memory returns to caching allocator.
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

        // NOTE: pending commit from the previous step is resolved lazily:
        // - Super fast path: resolved AFTER graph launch (overlaps with GPU)
        // - Normal path: resolved below, before prepare_inputs needs it

        // Clean up finished requests.
        let has_preempted = scheduler_output
            .preempted_req_ids
            .as_ref()
            .is_some_and(|s| !s.is_empty());
        self.batch_changed = !scheduler_output.finished_req_ids.is_empty()
            || !scheduler_output.scheduled_new_reqs.is_empty()
            || has_preempted
            || !scheduler_output
                .scheduled_cached_reqs
                .resumed_req_ids
                .is_empty();
        if self.batch_changed {
            // Batch composition changed — can't reuse persistent input_ids or metadata.
            self.last_graph_batch_size = None;
            self.graph_metadata_valid = false;
        }
        for req_id in &scheduler_output.finished_req_ids {
            self.token_buffers.remove(req_id);
            self.prompt_lengths.remove(req_id);
            self.annotation_buffers.remove(req_id);
            self.mm_data_buffers.remove(req_id);
            self.sampling_params_map.remove(req_id);
            self.seeded_rngs.remove(req_id);
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
                self.ensure_grammar_factory();
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

            tracing::info!(
                req_id = %new_req.req_id,
                prompt_len = prompt_ids.len(),
                cached = start,
                new = tokens_to_use.len(),
                hit_rate = format_args!("{:.0}%", if prompt_ids.is_empty() { 0.0 } else { start as f64 / prompt_ids.len() as f64 * 100.0 }),
                "KV cache hit",
            );

            self.token_buffers
                .insert(new_req.req_id.clone(), prompt_ids.to_vec());
            self.prompt_lengths
                .insert(new_req.req_id.clone(), prompt_ids.len());
            if let Some(ref ann) = new_req.block_annotations {
                self.annotation_buffers
                    .insert(new_req.req_id.clone(), ann.clone());
            }
            if let Some(ref mm) = new_req.mm_data {
                self.mm_data_buffers
                    .insert(new_req.req_id.clone(), mm.clone());
            }
            if let Some(ref params) = new_req.sampling_params {
                self.sampling_params_map
                    .insert(new_req.req_id.clone(), params.clone());
                // Create per-request seeded RNG if seed is specified.
                if let Some(seed) = params.seed {
                    use rand::SeedableRng;
                    self.seeded_rngs.insert(
                        new_req.req_id.clone(),
                        rand::rngs::StdRng::seed_from_u64(seed),
                    );
                }
                #[cfg(feature = "guided-decoding")]
                if let Some(ref grammar) = params.guided_grammar
                    && let Some(ref factory) = self.grammar_factory
                {
                    match vllm_model::grammar::GrammarGuide::from_guided_grammar(grammar, factory) {
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

        // ---------------------------------------------------------------
        // PP token fixup: when the scheduler provides new_token_ids (PP sync
        // scheduling), overwrite last_token_ids in the input batch so that
        // non-last PP stages embed the correct token instead of the dummy 0
        // committed in the previous step.
        // Matches Python: gpu_model_runner.py _update_states lines 1143-1150.
        // ---------------------------------------------------------------
        let cached = &scheduler_output.scheduled_cached_reqs;
        if !cached.new_token_ids.is_empty() {
            for (i, req_id) in cached.req_ids.iter().enumerate() {
                if let Some(tokens) = cached.new_token_ids.get(i)
                    && let Some(&last_token) = tokens.last()
                {
                    self.input_batch.set_last_token(req_id, last_token);
                }
            }
        }

        // ---------------------------------------------------------------
        // Chunked prefill re-arm: detect cached requests that need
        // more than 1 token (prefill continuation). Must happen before
        // the super fast graph path, which would otherwise replay a
        // decode graph instead of running the prefill eagerly.
        // ---------------------------------------------------------------
        let has_chunked_prefill =
            scheduler_output
                .scheduled_cached_reqs
                .req_ids
                .iter()
                .any(|req_id| {
                    let is_resumed = scheduler_output
                        .scheduled_cached_reqs
                        .resumed_req_ids
                        .contains(req_id);
                    let num_scheduled = scheduler_output
                        .num_scheduled_tokens
                        .get(req_id)
                        .copied()
                        .unwrap_or(0);
                    is_resumed || num_scheduled > 1
                });

        // ---------------------------------------------------------------
        // Super fast path: skip prepare_inputs entirely when the graph
        // has valid metadata from the previous step. This avoids ~50μs
        // of CPU work and, critically, lets us defer commit_step(N-1)
        // to AFTER the graph launch so it overlaps with GPU execution.
        // ---------------------------------------------------------------
        let num_active = self.input_batch.num_active();
        let fast_graph_bs = if self.graph_metadata_valid && !has_chunked_prefill {
            self.graph_runner
                .as_ref()
                .and_then(|r| r.nearest_graph_size(num_active))
                .filter(|&gbs| self.last_graph_batch_size == Some(gbs))
        } else {
            None
        };

        if let Some(graph_bs) = fast_graph_bs
            && let Some(ref mut stg) = self.host_staging
            && let Some(ref mut device) = self.device
        {
            // Collect info from InputBatch upfront (immutable borrow ends here).
            let (req_ids, block_tables, tokens_in_pool) = self.input_batch.fast_path_info();
            let out_req_ids: Vec<String> = req_ids.to_vec();
            let token_counts = self.input_batch.fast_path_token_counts();
            // Max seqlen_k for this step = max(tokens_already_in_cache + 1).
            // The +1 accounts for the token we're about to decode (written
            // to cache by the fused QKV+rope+cache kernel before attention).
            let fast_max_seqlen_k = tokens_in_pool.iter().copied().max().unwrap_or(0) + 1;

            // Check all-greedy and no-logprobs/grammar without prepare_inputs.
            let all_greedy_fast = out_req_ids.iter().all(|rid| {
                self.sampling_params_map
                    .get(rid)
                    .is_none_or(|p| p.temperature < 1e-6)
            });
            let any_needs_full = self.seal_pad_processor.is_active()
                || out_req_ids.iter().any(|rid| {
                    self.sampling_params_map
                        .get(rid)
                        .is_some_and(|p| p.logprobs.is_some())
                        || {
                            #[cfg(feature = "guided-decoding")]
                            {
                                self.grammar_states.contains_key(rid)
                            }
                            #[cfg(not(feature = "guided-decoding"))]
                            {
                                false
                            }
                        }
                });

            if all_greedy_fast && !any_needs_full {
                let block_size = self.config.block_size;

                // Block table update for the graph (only if blocks changed).
                let new_bt = if blocks_changed {
                    let bt = unsafe { stg.fill_block_table(block_tables, graph_bs) };
                    Some(bt)
                } else {
                    None
                };

                // Graph launch — GPU self-updates positions, slot_mapping, seqused_k.
                let runner = self.graph_runner.as_ref().unwrap();
                let replay_out = unsafe {
                    runner.replay_decode_fast(
                        graph_bs,
                        None,
                        new_bt,
                        block_size,
                        fast_max_seqlen_k,
                        device,
                    )
                }
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("super fast replay_decode_fast: {e}"))
                })?;

                // Async D2H — enqueue on transfer stream, don't block.
                let buf_idx = stg.token_buf_idx;
                Self::d2h_token_ids_async(stg, buf_idx, &replay_out.token_ids, num_active, device)?;

                // NOW resolve the pending commit from the previous step.
                // The GPU is running step N, so this CPU work overlaps with it.
                if let Some(pending) = self.pending_commit.take() {
                    device.sync_d2h().map_err(|e| {
                        ExecutorError::WorkerExecution(format!("pending sync_d2h: {e}"))
                    })?;
                    let prev_ids = unsafe {
                        stg.host_token_ids[pending.buf_idx].slice::<u32>(pending.num_reqs)
                    };
                    for (i, &tok) in prev_ids.iter().enumerate().take(pending.num_reqs) {
                        self.input_batch.commit_step(
                            &pending.req_ids[i],
                            &[tok],
                            pending.token_counts[i],
                            pending.has_spec_tokens[i],
                        );
                        if let Some(buf) = self.token_buffers.get_mut(&pending.req_ids[i]) {
                            buf.push(tok);
                        }
                    }
                }

                self.pending_commit = Some(PendingCommit {
                    buf_idx,
                    num_reqs: num_active,
                    req_ids: out_req_ids.clone(),
                    token_counts,
                    has_spec_tokens: vec![false; num_active],
                });

                // Toggle double-buffer.
                stg.token_buf_idx ^= 1;

                // Build deferred output.
                let event_addr = device.d2h_done as usize;
                let buf_addr = stg.host_token_ids[buf_idx].ptr() as usize;
                let nr = num_active;
                return Ok(ModelRunnerOutput::deferred(
                    out_req_ids,
                    Box::new(move || unsafe {
                        driver::event_synchronize_raw(event_addr).expect("D2H event sync failed");
                        std::slice::from_raw_parts(buf_addr as *const u32, nr).to_vec()
                    }),
                ));
            }
        }

        // Resolve any deferred D2H commit from the previous step before
        // prepare_inputs (which reads positions, tokens_in_pool, last_token_ids).
        if let Some(pending) = self.pending_commit.take() {
            if let Some(ref dev) = self.device {
                dev.sync_d2h().map_err(|e| {
                    ExecutorError::WorkerExecution(format!("pending sync_d2h: {e}"))
                })?;
            }
            if let Some(ref stg) = self.host_staging {
                let host_ids =
                    unsafe { stg.host_token_ids[pending.buf_idx].slice::<u32>(pending.num_reqs) };
                for (i, &tok) in host_ids.iter().enumerate().take(pending.num_reqs) {
                    self.input_batch.commit_step(
                        &pending.req_ids[i],
                        &[tok],
                        pending.token_counts[i],
                        pending.has_spec_tokens[i],
                    );
                    if let Some(buf) = self.token_buffers.get_mut(&pending.req_ids[i]) {
                        buf.push(tok);
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Preemption/resumption fixup — done AFTER pending_commit resolution
        // so that token_buffers has the final token from the previous step
        // before we rebuild the prefill sequence.
        //
        // Preempted requests: remove from InputBatch to stop zombie GPU writes
        // into blocks that have been freed and reallocated.  We keep
        // token_buffers / sampling_params_map / seeded_rngs so the request
        // can be cleanly re-admitted.
        //
        // Resumed requests: re-add to InputBatch as a fresh prefill using the
        // complete token sequence (prompt + all output tokens generated so
        // far).  The pending_commit above has already appended the very last
        // token to token_buffers, so the sequence is complete.
        //
        // Matches Python gpu_model_runner._update_states: unscheduled requests
        // are removed, resumed requests are re-added via add_request.
        // ---------------------------------------------------------------
        if let Some(preempted) = &scheduler_output.preempted_req_ids {
            for req_id in preempted {
                self.input_batch.remove_request(req_id);
            }
        }
        for (i, req_id) in scheduler_output
            .scheduled_cached_reqs
            .req_ids
            .iter()
            .enumerate()
        {
            let is_resumed = scheduler_output
                .scheduled_cached_reqs
                .resumed_req_ids
                .contains(req_id);
            // Chunked prefill continuation: cached request with num_scheduled > 1
            // means the scheduler is sending another chunk of prompt tokens.
            // We must re-arm the InputBatch slot as prefill (remove + add_request)
            // so prepare_inputs emits the full chunk instead of a single decode token.
            let num_scheduled_for_req = scheduler_output
                .num_scheduled_tokens
                .get(req_id)
                .copied()
                .unwrap_or(0);
            // Only treat as chunked prefill if the request still has prompt tokens
            // remaining. A cached request with num_scheduled > 1 that has already
            // completed its prefill is just a normal decode (e.g. spec decode or
            // scheduler batching artifact) and must NOT be re-armed.
            let prompt_len = self.prompt_lengths.get(req_id).copied().unwrap_or(0);
            let num_computed_here = scheduler_output
                .scheduled_cached_reqs
                .num_computed_tokens
                .get(i)
                .copied()
                .unwrap_or(0) as usize;
            let is_chunked_prefill_continuation = !is_resumed
                && num_scheduled_for_req > 1
                && num_computed_here + num_scheduled_for_req < prompt_len;
            if !is_resumed && !is_chunked_prefill_continuation {
                continue;
            }
            // Re-add the resumed/chunked-prefill request as a fresh prefill.
            let new_block_ids: Vec<usize> = scheduler_output
                .scheduled_cached_reqs
                .new_block_ids
                .get(i)
                .and_then(|opt| opt.as_ref())
                .and_then(|groups| groups.first())
                .cloned()
                .unwrap_or_default();
            let num_computed = scheduler_output
                .scheduled_cached_reqs
                .num_computed_tokens
                .get(i)
                .copied()
                .unwrap_or(0);
            // Truncate to the scheduled chunk: scheduler allocated blocks for
            // exactly `num_scheduled_tokens` tokens starting from `num_computed`.
            // Passing the full token_buffers (prompt + all outputs) would give
            // seq_lens > available blocks → slot_mapping = -1 → GPU fault.
            // This mirrors the new-request chunked-prefill logic (lines ~4906-4908).
            let num_scheduled = scheduler_output
                .num_scheduled_tokens
                .get(req_id)
                .copied()
                .unwrap_or(0);
            let tokens: Vec<u32> = self
                .token_buffers
                .get(req_id)
                .map(|buf| {
                    let start = num_computed as usize;
                    let end = (start + num_scheduled).min(buf.len());
                    buf[start..end].to_vec()
                })
                .unwrap_or_default();
            // For both resumed and chunked prefill, the scheduler's
            // allocate_slots returns ALL block IDs (not a delta).
            // For chunked prefill, update_blocks already set the full
            // block table on the input_batch slot — grab it before remove.
            let all_block_ids = if is_chunked_prefill_continuation {
                // update_blocks (line ~6359) already set the full block table
                self.input_batch
                    .block_table(req_id)
                    .map(|b| b.to_vec())
                    .unwrap_or(new_block_ids)
            } else {
                new_block_ids
            };
            // Remove the old (zombie) slot first, then re-add as prefill.
            self.input_batch.remove_request(req_id);
            self.input_batch
                .add_request(req_id.clone(), &tokens, all_block_ids, num_computed);
        }

        // Prepare flat inputs from InputBatch.
        let prepared = self
            .input_batch
            .prepare_inputs(&scheduler_output.scheduled_spec_decode_tokens);
        if prepared.flat_token_ids.is_empty() {
            return Ok(ModelRunnerOutput::from_token_map(HashMap::new()));
        }

        let total_tokens = prepared.flat_token_ids.len();
        let pp_active = self.pp_config.is_some_and(|pp| pp.pp_size > 1);

        // -------------------------------------------------------------------
        // Pipeline parallelism: non-last stages
        // Run eager forward, send intermediates, return dummy output.
        // Must be before the model/kv/device destructuring to avoid borrow conflicts.
        // -------------------------------------------------------------------
        #[cfg(feature = "nccl")]
        if pp_active {
            let pp = self.pp_config.unwrap();
            if !pp.is_last_stage() {
                return self.execute_pp_non_last_stage(pp, prepared, block_size);
            }
        }

        // -------------------------------------------------------------------
        // PP last stage: recv intermediates from previous stage before forward.
        // Must be before the model/kv/device destructuring to avoid borrow conflicts.
        // -------------------------------------------------------------------
        #[cfg(feature = "nccl")]
        let pp_intermediate: Option<(vllm_cuda::OwnedTensor, vllm_cuda::OwnedTensor)> = if pp_active
        {
            let pp = self.pp_config.unwrap();
            debug_assert!(pp.is_last_stage());
            if !pp.is_first_stage() {
                let pp_group = self.pp_group.as_ref().unwrap();
                let hs_buf = self.pp_recv_hs_buf.as_ref().unwrap();
                let res_buf = self.pp_recv_res_buf.as_ref().unwrap();
                let device = self.device.as_mut().ok_or_else(|| {
                    ExecutorError::WorkerExecution("device not initialized".into())
                })?;
                Some(Self::pp_recv_intermediates(
                    pp_group,
                    hs_buf,
                    res_buf,
                    pp,
                    total_tokens,
                    device,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        #[cfg(not(feature = "nccl"))]
        let pp_intermediate: Option<(vllm_cuda::OwnedTensor, vllm_cuda::OwnedTensor)> = None;

        // Spans: mark per-block rotation flags based on BlockAnnotations.
        //
        // Relocatable blocks go through a rotate-attend-unrotate cycle:
        // - K is written WITH RoPE (normal QKV projection)
        // - Post-attention: un-rotated (inverse RoPE) → position-independent
        // - Pre-attention (next step): rotated to current position
        //
        // `block_is_unrotated[physical_block] = true` means:
        //   "this Relocatable block's K is currently stored WITHOUT RoPE
        //    (from a prior step's post-attention un-rotation pass)."
        //
        // Freshly written blocks are false — K has RoPE from QKV projection.
        // Non-Relocatable blocks are always false (never touched).
        if !self.annotation_buffers.is_empty()
            && let Some(kv_cache) = self.kv_cache.as_mut()
        {
            let meta = &prepared.attn_meta;
            for i in 0..meta.num_reqs {
                let req_id = &meta.req_ids[i];
                if let Some(annotations) = self.annotation_buffers.get(req_id) {
                    let block_ids = &meta.block_ids[i];
                    let tokens_before = meta.tokens_before[i];
                    let seq_len = meta.seq_lens[i];
                    for (block_idx, &physical_block) in block_ids.iter().enumerate() {
                        let block_start_pos = block_idx * block_size;
                        if block_start_pos >= seq_len {
                            break; // past the end of the sequence
                        }
                        let (is_relocatable, is_unrotated) = vllm_common::compute_block_flags(
                            annotations,
                            block_idx,
                            block_size,
                            seq_len,
                            tokens_before,
                        );
                        kv_cache.mark_block(physical_block, is_relocatable, is_unrotated);
                    }
                }
            }
            // Upload flags to GPU for the attention kernel.
            unsafe {
                let stream = self
                    .device
                    .as_ref()
                    .map(|d| d.compute_stream)
                    .unwrap_or(std::ptr::null_mut());
                kv_cache.sync_block_flags_to_gpu(stream);
            }
        }

        // Compute once before the split borrow below (self is borrowed mutably for device).
        let max_blocks_per_seq = self.max_blocks_per_seq();

        // Split borrows: model + kv_cache (shared) vs device (mutable).
        // Use direct field access so the borrow checker sees disjoint borrows.
        // Note: gdn_pool_ref is deferred until after the piecewise path to avoid
        // borrow conflicts with execute_model_piecewise(&mut self).
        let (mut model, mut kv_cache, mut device) =
            match (&self.model, &self.kv_cache, &mut self.device) {
                (Some(m), Some(kv), Some(d)) => (m, kv, d),
                _ => {
                    return Err(ExecutorError::WorkerExecution(
                        "model, KV cache, or device not initialized".into(),
                    ));
                }
            };
        let num_reqs = prepared.req_inputs.len();

        // Build batch_req_ids for this step (used by processor updates after forward).
        self.batch_req_ids.clear();
        self.batch_req_ids
            .extend(prepared.req_inputs.iter().map(|r| r.req_id.clone()));

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
                    gpu_input_ids.view(),
                    gpu_positions.view(),
                    slot_mapping.view(),
                    cu_seqlens_q.view(),
                    seqused_k.view(),
                    block_table_gpu.view(),
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                    None,
                )
            };

            // Pool each request's hidden states slice.
            let strategy = self.pooling_strategy;
            let meta = &prepared.attn_meta;
            let mut pooler_map: HashMap<String, EmbeddingData> = HashMap::new();

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
        // PP last stage: skip CUDA graphs for now — use eager forward with forward_pp.
        let is_decode = prepared.attn_meta.q_lens.iter().all(|&q| q == 1);
        // MM-bearing reqs MUST take the eager path: the decode CUDA graph was
        // captured with 1D position tensors (single rope-kernel invocation
        // per layer); replaying it for an MM-bearing req would feed 1D
        // positions to the MRoPE kernel, which disagrees with the 3D
        // positions used to encode the cached visual KV → attention misfires
        // and the model emits `<|im_end|>` immediately (Bug 3, the symmetric
        // counterpart of Bug 1's prefill-graph gate at `use_prefill_graph`).
        // The eager else-branch below builds `[3, n_tokens]` positions via
        // `build_all_mm_position_patches` whether or not the encoder runs
        // this step.
        let any_mm_in_batch = prepared
            .req_inputs
            .iter()
            .any(|r| self.mm_data_buffers.contains_key(&r.req_id));
        let graph_bs = if pp_active {
            None // PP stages use eager forward only
        } else if is_decode && !any_mm_in_batch {
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
        let any_spec_in_batch = prepared
            .req_inputs
            .iter()
            .any(|r| !r.spec_token_ids.is_empty());
        let is_mixed = !is_decode && has_decode;
        let decode_graph_bs = if pp_active {
            None // PP stages use eager forward only
        } else if is_mixed && !any_spec_in_batch {
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
                let mut block_table = vec![0i32; decode_graph_bs * max_blocks_per_seq];

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
                        if j < max_blocks_per_seq {
                            block_table[out_idx * max_blocks_per_seq + j] = bid as i32;
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
                        self.config.block_size,
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

                let pf_meta = vllm_model::AttentionMetadata::new(
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
                    model.forward(
                        gpu_input_ids.view(),
                        gpu_positions.view(),
                        slot_mapping.view(),
                        cu_seqlens_q.view(),
                        seqused_k.view(),
                        block_table_gpu.view(),
                        max_seqlen_q,
                        max_seqlen_k,
                        kv_cache,
                        device,
                        last_token_indices.as_ref().map(|t| t.view()),
                        // Mixed-batch eager prefill: MM splice not yet plumbed
                        // through this slow path. Single-prefill flows go
                        // through the eager main-forward branch below where
                        // mm_inputs is built. A VL prefill that lands in the
                        // mixed path (rare — would require concurrent decode
                        // reqs at first-image-prefill step) is a follow-up.
                        None,
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
            // an OwnedTensor from forward(). merged_owned keeps the merged allocation.)
            let _ = decode_logits;
            drop(prefill_logits);

            // Fall through to sampling with merged logits.
            let logits = merged;

            // Update logits processors after forward, before sampling.
            Self::update_logits_processors(
                &LogitsUpdateCtx {
                    batch_changed: self.batch_changed,
                    num_reqs,
                    sampling_params_map: &self.sampling_params_map,
                    token_buffers: &self.token_buffers,
                    batch_req_ids: &self.batch_req_ids,
                },
                self.logits_pipeline.as_mut(),
                #[cfg(feature = "guided-decoding")]
                &mut self.grammar_states,
                &mut self.grammar_processor,
                &mut self.allowed_token_ids_processor,
                &mut self.seal_pad_processor,
                device,
            );

            // GPU sampling for mixed prefill+decode merged logits.
            return Self::gpu_sample_and_finalize(
                &self.sampling_params_map,
                #[cfg(feature = "guided-decoding")]
                &mut self.grammar_states,
                &self.grammar_processor,
                &self.allowed_token_ids_processor,
                &self.seal_pad_processor,
                self.logits_pipeline.as_ref(),
                &mut self.seeded_rngs,
                &self.host_staging,
                &mut self.input_batch,
                &mut self.token_buffers,
                &self.prompt_lengths,
                logits,
                prepared,
                device,
                all_greedy,
                vocab_size,
            );
        }

        // Check if any request needs logprobs, grammar, or logit processors
        // (these require the full GPU sampling pipeline instead of in-graph argmax).
        // Uses request-level state (sampling_params_map, grammar_states) rather than
        // processor GPU state, since processors are updated after forward/graph replay.
        let any_needs_full_sampling = prepared.req_inputs.iter().any(|r| {
            self.sampling_params_map.get(&r.req_id).is_some_and(|p| {
                p.logprobs.is_some()
                    || p.logit_bias.is_some()
                    || p.frequency_penalty != 0.0
                    || p.presence_penalty != 0.0
                    || p.repetition_penalty != 1.0
                    || p.min_tokens > 0
                    || p.bad_words_token_ids.is_some()
                    || p.allowed_token_ids.is_some()
                    || p.seal
            })
        }) || {
            #[cfg(feature = "guided-decoding")]
            {
                !self.grammar_states.is_empty()
            }
            #[cfg(not(feature = "guided-decoding"))]
            {
                false
            }
        };

        // -----------------------------------------------------------------------
        // Runtime Mode Selection: Choose between Full and Piecewise CUDA graphs
        // -----------------------------------------------------------------------
        let use_piecewise = if use_graph {
            // Determine if we should use piecewise mode for this batch
            match self.config.cuda_graph_mode {
                CudaGraphMode::None => false,
                CudaGraphMode::Full => false,
                CudaGraphMode::Piecewise => true,
                // For pure decode batches, FullAndPiecewise uses full graphs.
                // Piecewise is reserved for mixed/chunked-prefill batches (future).
                CudaGraphMode::FullAndPiecewise => false,
                CudaGraphMode::FullDecodeOnly => false,
                CudaGraphMode::Auto => {
                    unreachable!("Auto should be resolved in compile_or_warm_up_model")
                }
            }
        } else {
            false
        };

        // -----------------------------------------------------------------------
        // Piecewise CUDA Graph Path: Dynamic attention with optimal split-K
        // -----------------------------------------------------------------------
        if use_piecewise && use_graph && all_greedy && !any_needs_full_sampling {
            let graph_bs = graph_bs.unwrap();

            // Validate piecewise runner is initialized
            if self.piecewise_graph_runner.is_none() {
                tracing::warn!(
                    "Piecewise mode requested but runner not initialized, falling back to full mode"
                );
            } else {
                tracing::debug!(
                    "Using piecewise CUDA graph for decode batch size {}",
                    num_reqs
                );

                // Drop borrowed fields before calling execute_model_piecewise, which needs &mut self.
                let _ = model;
                let _ = kv_cache;
                let _ = device;

                // Execute piecewise replay flow
                let result = unsafe { self.execute_model_piecewise(&prepared) };

                match result {
                    Ok(host_ids) => {
                        // Commit step and build output.
                        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
                            let tok = host_ids[req_idx];
                            self.input_batch.commit_step(
                                &req_slice.req_id,
                                &[tok],
                                req_slice.token_count,
                                !req_slice.spec_token_ids.is_empty(),
                            );
                            if let Some(buf) = self.token_buffers.get_mut(&req_slice.req_id) {
                                buf.push(tok);
                            }
                        }
                        let req_ids: Vec<String> = prepared
                            .req_inputs
                            .iter()
                            .map(|r| r.req_id.clone())
                            .collect();
                        self.input_batch.reclaim_buffers(prepared);
                        // Success - return piecewise output
                        self.last_graph_batch_size = Some(graph_bs);
                        self.graph_metadata_valid = false; // Piecewise doesn't support fast path yet
                        return Ok(ModelRunnerOutput::from_ordered(req_ids, host_ids));
                    }
                    Err(e) => {
                        tracing::error!(
                            "Piecewise execution failed: {}, falling back to full mode",
                            e
                        );
                        // Fall through to full mode — re-obtain device borrow.
                    }
                }
                // Re-obtain borrows for fallback paths.
                model = self.model.as_ref().unwrap();
                kv_cache = self.kv_cache.as_ref().unwrap();
                device = self.device.as_mut().unwrap();
            }
        }

        // Deferred: take gdn_pool_ref now that piecewise path (which needs &mut self) is done.
        let gdn_pool_ref = self.gdn_state_pool.as_ref();

        // -----------------------------------------------------------------------
        // Full CUDA Graph Path: Monolithic graph with fixed split-K
        // -----------------------------------------------------------------------
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

                let max_seqlen_k_step = meta.seq_lens.iter().copied().max().unwrap_or(0);
                let runner = self.graph_runner.as_ref().unwrap();
                unsafe {
                    runner.replay_decode_fast(
                        graph_bs,
                        None, // input_ids already scattered by previous graph
                        new_bt,
                        block_size,
                        max_seqlen_k_step,
                        device,
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
                        block_size,
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

                let mut block_table = vec![0i32; graph_bs * max_blocks_per_seq];
                for (i, blocks) in meta.block_ids.iter().enumerate() {
                    for (j, &bid) in blocks.iter().enumerate() {
                        if j < max_blocks_per_seq {
                            block_table[i * max_blocks_per_seq + j] = bid as i32;
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
                        block_size,
                        device,
                        skip_input_ids,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("graph replay: {e}")))?
            };

            // Record that graph buffers now have valid metadata for next step.
            self.last_graph_batch_size = Some(graph_bs);
            self.graph_metadata_valid = true;

            // Deferred D2H: enqueue async copy on transfer stream, return
            // immediately without blocking on the GPU. The token IDs are
            // resolved lazily by the main thread (via D2HResolver) and the
            // commit_step is deferred to the start of the next execute_model.
            if let Some(ref mut stg) = self.host_staging {
                let buf_idx = stg.token_buf_idx;
                Self::d2h_token_ids_async(stg, buf_idx, &replay_out.token_ids, num_reqs, device)?;

                // Capture info needed for deferred commit_step.
                let req_ids: Vec<String> = prepared
                    .req_inputs
                    .iter()
                    .map(|r| r.req_id.clone())
                    .collect();
                let token_counts: Vec<usize> =
                    prepared.req_inputs.iter().map(|r| r.token_count).collect();
                let has_spec_tokens: Vec<bool> = prepared
                    .req_inputs
                    .iter()
                    .map(|r| !r.spec_token_ids.is_empty())
                    .collect();

                self.input_batch.reclaim_buffers(prepared);

                self.pending_commit = Some(PendingCommit {
                    buf_idx,
                    num_reqs,
                    req_ids: req_ids.clone(),
                    token_counts,
                    has_spec_tokens,
                });

                // Toggle double-buffer for next step.
                stg.token_buf_idx ^= 1;

                // Build deferred output: the resolver closure syncs the D2H
                // event and reads token IDs from the pinned host buffer.
                // Cast raw pointers to usize for Send safety (pinned buffer
                // and event outlive the closure — see PendingCommit safety doc).
                let event_addr = device.d2h_done as usize;
                let buf_addr = stg.host_token_ids[buf_idx].ptr() as usize;
                return Ok(ModelRunnerOutput::deferred(
                    req_ids,
                    Box::new(move || unsafe {
                        driver::event_synchronize_raw(event_addr).expect("D2H event sync failed");
                        std::slice::from_raw_parts(buf_addr as *const u32, num_reqs).to_vec()
                    }),
                ));
            }

            // No pinned staging — fall back to synchronous D2H.
            return Self::finalize_d2h_and_commit(
                &replay_out.token_ids,
                num_reqs,
                prepared,
                device,
                &self.host_staging,
                0,
                &mut self.input_batch,
                &mut self.token_buffers,
                &self.prompt_lengths,
            );
        }

        // Non-greedy graph path or eager path: need separate sampling.
        // IMPORTANT: skip graph replay when any logits processor or grammar is
        // active. CUDA graphs replay kernels at captured memory addresses —
        // intermediate forward-pass buffers reuse the same addresses as during
        // `_logits_owned` keeps the OwnedTensor alive for eager-forward paths
        // so the GPU memory backing `logits` survives until sampling completes.
        // `_mm_holder` keeps the vision-encoder OwnedTensor alive for the
        // same reason on the MM splice path (text-only batches: stays None).
        let mut _mm_holder: Option<(OwnedTensor, Vec<ferrite_forward::EmbedPatch>)> = None;
        let (_logits_owned, logits) = if use_graph {
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

                let max_seqlen_k_step = meta.seq_lens.iter().copied().max().unwrap_or(0);
                let runner = self.graph_runner.as_ref().unwrap();
                unsafe {
                    runner.replay_decode_fast(
                        graph_bs,
                        Some(input_ids_slice),
                        new_bt,
                        block_size,
                        max_seqlen_k_step,
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
                        block_size,
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

                let mut block_table = vec![0i32; graph_bs * max_blocks_per_seq];
                for (i, blocks) in meta.block_ids.iter().enumerate() {
                    for (j, &bid) in blocks.iter().enumerate() {
                        if j < max_blocks_per_seq {
                            block_table[i * max_blocks_per_seq + j] = bid as i32;
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
                        block_size,
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
            let logits = if graph_bs > num_reqs {
                replay_out.logits.narrow_dim0(0, num_reqs)
            } else {
                replay_out.logits
            };
            (None, logits)
        } else {
            // Non-decode path: try prefill graph, fall back to eager.
            self.last_graph_batch_size = None;
            self.graph_metadata_valid = false;

            // Check if we can use a prefill graph: single request, fresh prefill
            // (tokens_before == 0 means q_len == seq_len, so the model uses contiguous
            // FA2 — not paged — which is safe to capture in a CUDA graph).
            // MM-bearing reqs MUST take the eager path: the captured prefill graph
            // does not include the vision encoder + Embed splice, so replaying it
            // would skip vision_forward entirely and hand the decoder placeholder
            // tokens with no visual content (hallucinated output).
            let meta = &prepared.attn_meta;
            let req_has_mm = self.mm_data_buffers.contains_key(&meta.req_ids[0]);
            let use_prefill_graph = num_reqs == 1
                && meta.tokens_before[0] == 0
                && !req_has_mm
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
                let mut block_table = vec![0i32; max_blocks_per_seq];
                for (j, &bid) in meta.block_ids[0].iter().enumerate() {
                    if j < max_blocks_per_seq {
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
                        block_size,
                        device,
                    )
                }
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("prefill graph replay: {e}"))
                })?;

                (None, replay_out.logits)
            } else {
                // Eager forward path (multi-request prefill or uncaptured size).
                // Caching allocator: no reset needed — tensors freed on drop.

                // DEBUG: sync before forward to isolate errors from previous steps.
                /* {
                    let any_spec = prepared
                        .req_inputs
                        .iter()
                        .any(|r| !r.spec_token_ids.is_empty());
                    if any_spec {
                        unsafe { vllm_cuda::driver::stream_synchronize(device.compute_stream) }
                            .map_err(|e| {
                                ExecutorError::WorkerExecution(format!(
                                    "eager pre-forward sync (spec batch): {e}"
                                ))
                            })?;
                        let meta = &prepared.attn_meta;
                        tracing::warn!(
                            "spec decode eager forward: num_reqs={}, total_tokens={}, q_lens={:?}, \
                             seq_lens={:?}, tokens_before={:?}, block_ids_lens={:?}",
                            meta.num_reqs,
                            meta.total_tokens,
                            meta.q_lens,
                            meta.seq_lens,
                            meta.tokens_before,
                            meta.block_ids.iter().map(|b| b.len()).collect::<Vec<_>>(),
                        );
                    }
                } */

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

                // For spec decode: skip last_token_indices so we get logits
                // for ALL token positions (needed for rejection sampling).
                // For normal batches: gather only the last token per request.
                let any_spec_decode = prepared
                    .req_inputs
                    .iter()
                    .any(|r| !r.spec_token_ids.is_empty());

                let last_token_indices = if any_spec_decode {
                    // Spec decode: need logits for all positions, not just last.
                    // Model returns [total_tokens, vocab_size].
                    None
                } else if num_reqs < total_tokens {
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

                if model.is_qwen3_next() {
                    // Build GDN forward context. Covers both the
                    // hand-written `CudaModel::Qwen3Next` legacy path
                    // and ferrite-loaded `qwen3_next` arches; the
                    // dispatch inside `forward_qwen3_next` routes
                    // accordingly.
                    let gdn_pool = gdn_pool_ref.unwrap();
                    let meta = &prepared.attn_meta;

                    // Clear GDN state for sequences in prefill (query_len > 1).
                    for i in 0..meta.num_reqs {
                        let qlen = meta.query_start_loc[i + 1] - meta.query_start_loc[i];
                        if qlen > 1 {
                            unsafe { gdn_pool.clear_slot(i, device.compute_stream) }.map_err(
                                |e| ExecutorError::WorkerExecution(format!("GDN clear_slot: {e}")),
                            )?;
                        }
                    }

                    let (gdn_state_indices, gdn_cu_seqlens, num_seqs) =
                        Self::build_gdn_tensors(meta, device)?;
                    let logits = unsafe {
                        model.forward_qwen3_next(
                            gpu_input_ids.view(),
                            gpu_positions.view(),
                            slot_mapping.view(),
                            cu_seqlens_q.view(),
                            seqused_k.view(),
                            block_table.view(),
                            max_seqlen_q,
                            max_seqlen_k,
                            kv_cache,
                            gdn_pool,
                            gdn_state_indices.view(),
                            gdn_cu_seqlens.view(),
                            num_seqs,
                            device,
                            last_token_indices.as_ref().map(|t| t.view()),
                        )
                    };
                    let logits_gpu = *logits;
                    (Some(logits), logits_gpu)
                } else if pp_active {
                    // PP last stage: use forward_pp with received intermediates.
                    let input_ids = if self.pp_config.unwrap().is_first_stage() {
                        Some(gpu_input_ids.view())
                    } else {
                        None
                    };
                    let result = unsafe {
                        model.forward_pp(
                            input_ids,
                            pp_intermediate,
                            gpu_positions.view(),
                            slot_mapping.view(),
                            cu_seqlens_q.view(),
                            seqused_k.view(),
                            block_table.view(),
                            max_seqlen_q,
                            max_seqlen_k,
                            kv_cache,
                            device,
                            last_token_indices.as_ref().map(|t| t.view()),
                        )
                    };
                    let owned = match result {
                        vllm_cuda::model::llama::ForwardOutput::Logits(t) => t,
                        vllm_cuda::model::llama::ForwardOutput::Intermediate { .. } => {
                            unreachable!("last PP stage should return Logits");
                        }
                    };
                    let logits = *owned;
                    (Some(owned), logits)
                } else {
                    // Run vision encoder for any MM-bearing req at its
                    // first prefill step; output rides into model.forward
                    // via mm_inputs and the `Instruction::Embed::eval`
                    // splice. The OwnedTensor lands in `_mm_holder` so
                    // its caching-allocator slot stays live until end of
                    // the surrounding fn — the splice D2D-copies on
                    // `device.compute_stream` and a parallel allocation
                    // could otherwise alias the same memory region
                    // before the splice kernel actually runs.
                    _mm_holder = unsafe {
                        Self::run_mm_vision_forward(model, &self.mm_data_buffers, &prepared, device)
                    };
                    // For image-bearing batches on MRoPE arches, override
                    // the 1D `gpu_positions` with a `[3, n_tokens]` u32
                    // tensor where image tokens carry their (T,H,W) grid
                    // coordinates (Python parity with
                    // `Qwen2VLForConditionalGeneration.get_input_positions_tensor`).
                    //
                    // Per-req seq-space MM info covers EVERY MM-bearing
                    // req in the batch — including ones with
                    // `tokens_before > 0` (cached prefix) where the
                    // encoder didn't run this step. Load-bearing for the
                    // cached-prefix path: the MRoPE positions written
                    // for the trailing new tokens must match the cursor
                    // the encoder advanced to when it wrote the cached
                    // KV blocks, otherwise attention misfires (Bug 3).
                    // Text-only batches get an empty outer `Vec` and keep
                    // the pre-built 1D `gpu_positions` (broadcast path).
                    let per_req_mm =
                        Self::build_per_req_mm_seq_info(model, &self.mm_data_buffers, &prepared);
                    let gpu_positions_2d = if !per_req_mm.is_empty() {
                        let pos_2d = Self::build_mrope_positions_2d(&prepared, &per_req_mm);
                        let mut t = Self::h2d_u32(&pos_2d, device)?;
                        unsafe { t.reshape(&[3, total_tokens], GpuDType::U32) };
                        Some(t)
                    } else {
                        None
                    };
                    let positions_view = match &gpu_positions_2d {
                        Some(t) => t.view(),
                        None => gpu_positions.view(),
                    };
                    let mm_inputs: Option<MmForwardInputs<'_>> =
                        _mm_holder.as_ref().map(|(t, p)| MmForwardInputs {
                            mm_embeds: t.view(),
                            embed_patches: p.as_slice(),
                        });
                    let owned = unsafe {
                        model.forward(
                            gpu_input_ids.view(),
                            positions_view,
                            slot_mapping.view(),
                            cu_seqlens_q.view(),
                            seqused_k.view(),
                            block_table.view(),
                            max_seqlen_q,
                            max_seqlen_k,
                            kv_cache,
                            device,
                            last_token_indices.as_ref().map(|t| t.view()),
                            mm_inputs.as_ref(),
                        )
                    };
                    drop(gpu_positions_2d);
                    let logits = *owned;
                    (Some(owned), logits)
                }
            }
        };

        // DEBUG: sync after forward to catch forward errors.
        /* {
            let any_spec = prepared
                .req_inputs
                .iter()
                .any(|r| !r.spec_token_ids.is_empty());
            if any_spec {
                unsafe { vllm_cuda::driver::stream_synchronize(device.compute_stream) }.map_err(
                    |e| {
                        ExecutorError::WorkerExecution(format!(
                            "eager post-forward sync (spec batch): {e}"
                        ))
                    },
                )?;
            }
        } */

        // _logits_owned (if Some) keeps the OwnedTensor alive until after sampling.
        // It will be dropped at the end of this function, returning memory to the
        // caching allocator.

        // Update logits processors after forward/graph replay, before sampling.
        // This ensures processor GPU tensor allocations don't collide with CUDA
        // graph intermediate addresses.
        Self::update_logits_processors(
            &LogitsUpdateCtx {
                batch_changed: self.batch_changed,
                num_reqs,
                sampling_params_map: &self.sampling_params_map,
                token_buffers: &self.token_buffers,
                batch_req_ids: &self.batch_req_ids,
            },
            self.logits_pipeline.as_mut(),
            #[cfg(feature = "guided-decoding")]
            &mut self.grammar_states,
            &mut self.grammar_processor,
            &mut self.allowed_token_ids_processor,
            &mut self.seal_pad_processor,
            device,
        );

        // GPU sampling: handles all cases — greedy, non-greedy, penalties,
        // grammar, logit_bias, logprobs — entirely on GPU. No CPU fallback.
        Self::gpu_sample_and_finalize(
            &self.sampling_params_map,
            #[cfg(feature = "guided-decoding")]
            &mut self.grammar_states,
            &self.grammar_processor,
            &self.allowed_token_ids_processor,
            &self.seal_pad_processor,
            self.logits_pipeline.as_ref(),
            &mut self.seeded_rngs,
            &self.host_staging,
            &mut self.input_batch,
            &mut self.token_buffers,
            &self.prompt_lengths,
            logits,
            prepared,
            device,
            all_greedy,
            vocab_size,
        )
    }
    /// Update logits processor pipeline, grammar, and allowed_token_ids state.
    /// Called after forward/graph replay and before sampling, so processor GPU
    /// tensor allocations don't conflict with CUDA graph intermediate addresses
    /// (matching Python vLLM's architecture).
    fn update_logits_processors(
        ctx: &LogitsUpdateCtx<'_>,
        logits_pipeline: Option<&mut LogitsProcessorPipeline>,
        #[cfg(feature = "guided-decoding")] grammar_states: &mut HashMap<
            String,
            vllm_model::grammar::GrammarGuide,
        >,
        grammar_processor: &mut GrammarMaskProcessor,
        allowed_token_ids_processor: &mut AllowedTokenIdsProcessor,
        seal_pad_processor: &mut SealPadProcessor,
        device: &mut GpuDevice,
    ) {
        let batch_update = if ctx.batch_changed {
            Some(BatchUpdate {
                batch_size: ctx.num_reqs,
                added: Vec::new(),
                removed: Vec::new(),
            })
        } else {
            None
        };

        if let Some(pipeline) = logits_pipeline {
            pipeline.update_state(
                batch_update.as_ref(),
                ctx.sampling_params_map,
                ctx.token_buffers,
                ctx.batch_req_ids,
                device,
            );
        }

        #[cfg(feature = "guided-decoding")]
        {
            let grammar_reqs: Vec<(usize, Vec<u32>)> = ctx
                .batch_req_ids
                .iter()
                .enumerate()
                .filter_map(|(idx, rid)| {
                    grammar_states
                        .get_mut(rid)
                        .and_then(|g| g.allowed_tokens())
                        .map(|allowed| (idx, allowed))
                })
                .collect();
            let refs: Vec<(usize, &[u32])> = grammar_reqs
                .iter()
                .map(|(idx, v)| (*idx, v.as_slice()))
                .collect();
            grammar_processor.update_from_allowed_tokens(&refs, device);
        }
        #[cfg(not(feature = "guided-decoding"))]
        {
            let empty: Vec<(usize, &[u32])> = Vec::new();
            grammar_processor.update_from_allowed_tokens(&empty, device);
        }

        allowed_token_ids_processor.update_state(
            batch_update.as_ref(),
            ctx.sampling_params_map,
            ctx.token_buffers,
            ctx.batch_req_ids,
            device,
        );

        seal_pad_processor.update_state(
            batch_update.as_ref(),
            ctx.sampling_params_map,
            ctx.token_buffers,
            ctx.batch_req_ids,
            device,
        );
    }
} // end impl CudaWorker (execute_model_inner)

/// Read-only batch context passed to `update_logits_processors`.
struct LogitsUpdateCtx<'a> {
    batch_changed: bool,
    num_reqs: usize,
    sampling_params_map: &'a HashMap<String, SamplingParams>,
    token_buffers: &'a HashMap<String, Vec<u32>>,
    batch_req_ids: &'a [String],
}

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

    // =========================================================================
    // Piecewise CUDA Graph Tests
    // =========================================================================

    #[test]
    fn test_compute_optimal_splits_boundary_values() {
        // Test boundary: 0-256 should return 1 (key optimization)
        assert_eq!(compute_splits_logic(0), 1);
        assert_eq!(compute_splits_logic(1), 1);
        assert_eq!(compute_splits_logic(128), 1);
        assert_eq!(compute_splits_logic(256), 1);

        // Test boundary: 257-512 should return 2
        assert_eq!(compute_splits_logic(257), 2);
        assert_eq!(compute_splits_logic(384), 2);
        assert_eq!(compute_splits_logic(512), 2);

        // Test boundary: 513-1024 should return 4
        assert_eq!(compute_splits_logic(513), 4);
        assert_eq!(compute_splits_logic(768), 4);
        assert_eq!(compute_splits_logic(1024), 4);

        // Test boundary: 1025-2048 should return 8
        assert_eq!(compute_splits_logic(1025), 8);
        assert_eq!(compute_splits_logic(1536), 8);
        assert_eq!(compute_splits_logic(2048), 8);

        // Test boundary: >2048 should return 16
        assert_eq!(compute_splits_logic(2049), 16);
        assert_eq!(compute_splits_logic(4096), 16);
        assert_eq!(compute_splits_logic(8192), 16);
    }

    #[test]
    fn test_compute_optimal_splits_key_optimization() {
        // The key optimization: sequences ≤256 tokens use num_splits=1
        // This eliminates 72 kernel launches (36 transpose + 36 untranspose)
        for seqlen in [1, 64, 128, 192, 256] {
            assert_eq!(
                compute_splits_logic(seqlen),
                1,
                "Sequences ≤256 should use num_splits=1 to eliminate transpose overhead"
            );
        }

        // Verify that 257 triggers split-K
        assert_eq!(
            compute_splits_logic(257),
            2,
            "Sequences >256 should use split-K"
        );
    }

    #[test]
    fn test_compute_optimal_splits_progressive_scaling() {
        // Test that splits increase progressively with sequence length
        let splits_256 = compute_splits_logic(256);
        let splits_512 = compute_splits_logic(512);
        let splits_1024 = compute_splits_logic(1024);
        let splits_2048 = compute_splits_logic(2048);
        let splits_4096 = compute_splits_logic(4096);

        assert!(splits_256 <= splits_512);
        assert!(splits_512 <= splits_1024);
        assert!(splits_1024 <= splits_2048);
        assert!(splits_2048 <= splits_4096);

        // Verify specific values
        assert_eq!(splits_256, 1);
        assert_eq!(splits_512, 2);
        assert_eq!(splits_1024, 4);
        assert_eq!(splits_2048, 8);
        assert_eq!(splits_4096, 16);
    }

    #[test]
    fn test_compute_optimal_splits_power_of_two() {
        // All split values should be powers of 2 (1, 2, 4, 8, 16)
        for seqlen in [1, 100, 300, 600, 1200, 2400, 5000] {
            let splits = compute_splits_logic(seqlen);
            assert!(
                splits == 1 || splits == 2 || splits == 4 || splits == 8 || splits == 16,
                "Split value {} is not a power of 2 for seqlen {}",
                splits,
                seqlen
            );
        }
    }

    fn compute_splits_logic(max_seqlen_k: usize) -> usize {
        match max_seqlen_k {
            0..=256 => 1,
            257..=512 => 2,
            513..=1024 => 4,
            1025..=2048 => 8,
            _ => 16,
        }
    }
}
