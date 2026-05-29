// SPDX-License-Identifier: Apache-2.0
//! `FerriteWorker`: a `Worker` implementation backed by the ferrite GPU runtime.
//!
//! Cuda build: vllm-cuda + ferrite-forward (full feature set: CUDA graphs,
//! NCCL, FP8, etc.). Metal build: ferrite-forward MetalWorkerPool + the
//! ferrite-metal-kernels crate. The struct is shared with cfg-mutex'd fields.
//!
//! Gated behind the `cuda` or `metal` feature flag.

// Force the linker to keep `ferrite_models` — its `#[forward]`
// modules register with `inventory::submit!` for auto-discovery by
// `ferrite_forward::try_load`. Without this, the linker gc's the
// crate (no direct symbol references after the Phase B collapse)
// and the inventory comes up empty. Mirrored under metal now that
// the per-canonical metal `forward` body + `inventory::submit!`
// registration are emitted under `cfg(any(cuda, metal))` (Step 3.E).
#[cfg(any(feature = "cuda", feature = "metal"))]
extern crate ferrite_models as _;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tracing::info;
use vllm_common::SamplingParams;
#[cfg(feature = "cuda")]
use vllm_common::engine_io::EmbeddingData;
use vllm_config::CudaGraphMode;
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_engine::executor::ModelRunnerOutput;
use vllm_model::weight::HfModelConfig;

// Backend-neutral types lifted in Step 1 of the cfg-mutex extension:
// `KvCachePool` is `cfg(any(cuda, metal))` in `ferrite-kernels::kv_cache`,
// and `DType` is unconditional in `ferrite-cuda-core::dtype`.
use ferrite_cuda_core::dtype::DType as GpuDType;
use ferrite_kernels::kv_cache::KvCachePool;

#[cfg(feature = "cuda")]
use vllm_cuda::OwnedTensor;
#[cfg(feature = "cuda")]
use vllm_cuda::cpu_gpu_buf::PinnedBuf;
#[cfg(feature = "cuda")]
use vllm_cuda::device::GpuDevice;
#[cfg(feature = "cuda")]
use vllm_cuda::driver;
#[cfg(feature = "cuda")]
use vllm_cuda::graph::{CudaGraphRunner, PrefillGraphRunner};
#[cfg(feature = "cuda")]
// `graph_piece` is no longer wired in: the piecewise CUDA-graph capture path
// fed each `Self::Llama / Qwen2 / Gemma2 / Mixtral / …` arm by hand and only
// ran on the deleted `vllm_cuda::model::*` types. Ferrite-forward owns the
// forward and uses monolithic capture instead.
#[cfg(feature = "cuda")]
use vllm_cuda::logits_processor::{
    AllowedTokenIdsProcessor, BadWordsProcessor, BatchUpdate, GrammarMaskProcessor,
    LogitBiasProcessor, LogitsProcessor, LogitsProcessorPipeline, MinTokensProcessor,
    PenaltiesProcessor, SealPadProcessor,
};
#[cfg(feature = "cuda")]
use vllm_cuda::quant;
#[cfg(feature = "cuda")]
use vllm_cuda::tensor::{GpuTensor, TensorView};
#[cfg(feature = "cuda")]
use vllm_cuda::weights::GpuWeights;

// Backend-neutral types used by metal lifecycle bodies. Under cfg(metal),
// `GpuDevice` resolves to the Apple-silicon arm carrying `device + queue +
// allocator`; `OwnedTensor` / `TensorView` / `GpuTensor` / `GpuWeights` are
// the ones lifted to `cfg(any(cuda, metal))` in Step 1; `MetalAllocator`
// is the metal-side `BackendAllocator`; `RawGpuMem::from_buffer` wraps
// `metal::Buffer` for `KvCachePool::new`.
#[cfg(feature = "metal")]
use ::objc2_metal::{MTLBuffer as _, MTLDevice as _};
#[cfg(feature = "metal")]
use ferrite_cuda_core::weights::GpuWeights;
#[cfg(feature = "metal")]
use ferrite_cuda_core::{GpuDevice, GpuTensor, MetalAllocator, RawGpuMem, TensorView};

use crate::error::{ExecutorError, ExecutorResult};
use crate::input_batch::InputBatch;
#[cfg(feature = "cuda")]
use crate::input_batch::PreparedInputs;
use crate::worker::Worker;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a `FerriteWorker`.
#[derive(Debug, Clone)]
pub struct FerriteWorkerConfig {
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
    /// Optional draft-model path / HF repo ID for speculative decoding.
    /// `Some(_)` triggers a second model + KV pool load inside the same
    /// worker after the target loads (see
    /// `DRAFT_SPEC_DECODE_PLAN.md` phase 3). `None` = no draft.
    pub draft_model_path: Option<String>,
    /// Optional dtype override for the draft model's weights ("auto",
    /// "bfloat16", "float16", …). `None` inherits the target's dtype.
    /// Currently passed through but the metal path always coerces to
    /// bf16 to match the target — accepted for CLI parity with Python.
    pub draft_model_dtype: Option<String>,
}

/// Per-image MRoPE metadata in seq space, used by
/// [`FerriteWorker::build_mrope_positions_2d`] to walk the entire seq for
/// each MM-bearing req (including the cached-prefix portion) so the
/// cursor lands at the same position the encoder used to encode KV.
///
/// Tuple is `(seq_offset, length, grid_t, grid_h_merged, grid_w_merged)`
/// where `seq_offset` is relative to the req's full sequence start (NOT
/// batch start) — the same coordinate system as `PlaceholderRange.offset`.
#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
pub struct FerriteModel {
    pub weights: Box<dyn ferrite_forward::FerriteWeights>,
    /// Vision-encoder handle for multimodal arches. `Some(_)` only
    /// when the loaded checkpoint carries `visual.*` tensors AND the
    /// arch has submitted a `FerriteMmRegistration` row claiming the
    /// HF arch string. ferrite_worker calls
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

/// Loaded model — always a ferrite-forward-compiled arch.
///
/// Hand-written CUDA model forwards have been removed; every architecture
/// now flows through `ferrite_forward::try_load`. The single-variant enum
/// shape is preserved so existing match sites remain syntactically valid.
#[cfg(feature = "cuda")]
enum CudaModel {
    /// Any ferrite-compiled arch. Populated by
    /// `ferrite_forward::try_load(gw, stream, hf_arch_name)` —
    /// one variant covers every registered `#[forward]` module
    /// (llama / qwen2 / qwen3 / gemma2 / granite / …). Boxed
    /// because the per-arch `Weights` enum has one variant per
    /// compiled model config and can be large.
    Ferrite(Box<FerriteModel>),
}

#[cfg(feature = "cuda")]
impl CudaModel {
    fn num_layers(&self) -> usize {
        match self {
            Self::Ferrite(m) => m.weights.num_hidden_layers() as usize,
        }
    }

    /// Whether this model is a ferrite-compiled encoder (no lm_head Gemm,
    /// returns hidden states from `forward_backbone`). Encoders skip CUDA
    /// graph capture and use the fixed-estimate profiling path. Today only
    /// modernbert qualifies; add new arch names here as they land.
    fn is_ferrite_encoder(&self) -> bool {
        match self {
            Self::Ferrite(m) => matches!(m.weights.arch_name(), "modernbert"),
        }
    }

    fn num_kv_heads(&self) -> usize {
        match self {
            // FerriteWeights reports the unsharded config value;
            // divide by tp_world_size so KvCachePool gets the
            // per-rank head count, matching Python's
            // `max(1, total // tp_size)`.
            Self::Ferrite(m) => {
                let total = m.weights.num_key_value_heads() as usize;
                let tp = m.tp_world_size.max(1);
                (total / tp).max(1)
            }
        }
    }

    fn head_dim(&self) -> usize {
        match self {
            Self::Ferrite(m) => m.weights.head_dim() as usize,
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Ferrite(m) => m.weights.vocab_size() as usize,
        }
    }

    /// Whether this model uses Mixture-of-Experts layers.
    /// MoE models generate too many CUDA graph nodes for monolithic capture
    /// (router + N expert GEMMs + shared experts per layer) and exceed the
    /// CUDA driver's undocumented node limit. Piecewise capture is required.
    ///
    /// Identified by arch_name string; the set mirrors the previous
    /// hand-written `Mixtral / Qwen2Moe / Qwen3Moe / DeepSeekV2` enum check.
    fn is_moe(&self) -> bool {
        match self {
            Self::Ferrite(m) => matches!(
                m.weights.arch_name(),
                "mixtral"
                    | "qwen2_moe"
                    | "qwen3_moe"
                    | "deepseek_v2"
                    | "deepseek_v3"
                    | "deepseek_v3_flat"
            ),
        }
    }

    fn hidden_size(&self) -> usize {
        match self {
            Self::Ferrite(m) => m.weights.hidden_size() as usize,
        }
    }

    /// Inject NCCL process group into all model layers for TP.
    #[cfg(feature = "nccl")]
    fn set_tp_group(&mut self, group: std::sync::Arc<vllm_cuda::nccl::NcclGroup>) {
        match self {
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
                    vision_rope_cos: None,
                    vision_rope_sin: None,
                    pixels: None,
                    vision_cu_seqlens_full: None,
                    vision_cu_seqlens_window: None,
                    vision_max_seqlen_full: None,
                    vision_max_seqlen_window: None,
                    vision_window_index: None,
                    vision_reverse_indices: None,
                    vision_position_ids: None,
                    last_token_indices: None,
                    #[cfg(feature = "nccl")]
                    tp_group: m.tp_group.as_ref(),
                };
                m.weights.forward_backbone(&ctx, device, num_tokens)
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
            // Every ferrite-compiled arch routes through the
            // `FerriteWeights` trait's `forward` vtable. The per-arch
            // macro-emitted impl delegates to that arch's specialized
            // `forward(&weights, &ctx, device, num_tokens)`.
            //
            // Selective last-token gather for prefill: when
            // `last_token_indices` is set we plumb it through
            // `ForwardCtx.last_token_indices` so the lm_head op
            // (`Instruction::CutlassFusedAddRmsNormGemm`) gathers
            // [num_seqs, hidden] from the post-norm activations
            // BEFORE the GEMM. The post-forward gather then becomes
            // a no-op (idx.dim(0) == logits.dim(0)). Mirrors metal's
            // GatherLastToken lowering and Python vLLM's
            // `logits_indices = query_start_loc[1:] - 1`.
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
                    vision_rope_cos: None,
                    vision_rope_sin: None,
                    pixels: None,
                    vision_cu_seqlens_full: None,
                    vision_cu_seqlens_window: None,
                    vision_max_seqlen_full: None,
                    vision_max_seqlen_window: None,
                    vision_window_index: None,
                    vision_reverse_indices: None,
                    vision_position_ids: None,
                    last_token_indices: last_token_indices.as_ref().copied(),
                    #[cfg(feature = "nccl")]
                    tp_group: m.tp_group.as_ref(),
                };
                let logits = m.weights.forward(&ctx, device, num_tokens);
                // If the lm_head op already gathered (logits is at
                // [num_seqs, vocab]), skip the post-forward gather.
                // Otherwise (older codepaths / edge buckets that don't
                // hit the lm_head-with-indices arm) fall back to the
                // post-forward narrow.
                match last_token_indices {
                    Some(idx) if idx.dim(0) < num_tokens as usize && logits.dim(0) > idx.dim(0) => {
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
        }
    }
}

// ---------------------------------------------------------------------------
// Per-arch HF-config helpers (`llama_config_from_hf`, `mixtral_config_from_hf`,
// …, `deepseek_v2_config_from_hf`) used to live here. They built strongly-
// typed configs that fed the hand-written `vllm_cuda::model::*::*ForCausalLM`
// loaders. Both the loaders and these helpers are gone — ferrite-forward
// parses HF config inside its emitted per-arch `Weights::load` body.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Pinned host staging buffers
// ---------------------------------------------------------------------------

/// Pre-allocated pinned (page-locked) host buffers for CUDA graph replay.
///
/// Eliminates per-step heap Vec allocations and enables true async DMA
/// (pageable memory forces the CUDA driver to stage through an internal
/// pinned buffer, serializing the transfer).
#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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
// FerriteWorker
// ---------------------------------------------------------------------------

/// Deferred commit from a previous decode step.
///
/// Stored when the greedy graph fast path defers D2H sync. The token IDs
/// live in one of the double-buffered pinned host staging buffers. The
/// commit is resolved at the start of the next `execute_model_inner` call.
#[cfg(feature = "cuda")]
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

/// A worker backed by the ferrite GPU runtime (cuda or metal) for
/// zero-allocation inference.
pub struct FerriteWorker {
    // ---------------------------------------------------------------
    // Backend-neutral state (cfg-mutexed only by transitive types).
    // ---------------------------------------------------------------
    config: FerriteWorkerConfig,
    kv_cache: Option<KvCachePool>,
    model_dir: Option<PathBuf>,
    hf_config: Option<HfModelConfig>,
    /// Draft model's resolved snapshot dir, populated when
    /// `config.draft_model_path` is set. See
    /// `DRAFT_SPEC_DECODE_PLAN.md` phase 3.
    #[allow(dead_code)]
    draft_model_dir: Option<PathBuf>,
    /// Draft model's `config.json`, populated alongside `draft_model_dir`.
    #[allow(dead_code)]
    draft_hf_config: Option<HfModelConfig>,
    /// Second KV cache pool, sized for the draft model. Lives on the
    /// same `MetalAllocator` / residency set as the target's pool —
    /// one allocator covers all weights + both pools.
    #[allow(dead_code)]
    draft_kv_cache: Option<KvCachePool>,
    model_dtype: GpuDType,
    resolved_architecture: Option<String>,
    is_shutdown: bool,
    token_buffers: HashMap<String, Vec<u32>>,
    /// Per-request prompt length (for discard_request_mask on intermediate prefill chunks).
    prompt_lengths: HashMap<String, usize>,
    /// Per-request block annotations for span-aware RoPE.
    annotation_buffers: HashMap<String, vllm_common::BlockAnnotations>,
    /// Per-request multimodal data (images), only populated for requests
    /// scheduled at first as image-bearing.
    mm_data_buffers: HashMap<String, vllm_common::MultimodalData>,
    sampling_params_map: HashMap<String, SamplingParams>,
    input_batch: InputBatch,
    preloaded_tokenizer: Option<tokenizers::Tokenizer>,
    /// Resolved pooling strategy for embedding mode (cuda-only:
    /// pooling-mode codepaths live in the `cuda` impl block).
    #[cfg(feature = "cuda")]
    pooling_strategy: vllm_model::embedding::PoolingStrategy,
    /// Whether executing in pooling mode (--runner pooling).
    #[cfg(feature = "cuda")]
    is_pooling: bool,
    /// True if batch composition changed this step (triggers BatchUpdate).
    #[cfg(feature = "cuda")]
    batch_changed: bool,
    /// Ordered request IDs in the current batch (for pipeline update_state).
    #[cfg(feature = "cuda")]
    batch_req_ids: Vec<String>,
    /// Per-request seeded RNGs for deterministic sampling.
    seeded_rngs: HashMap<String, rand::rngs::StdRng>,
    /// Optional progress callback for startup initialization.
    #[allow(clippy::type_complexity)]
    progress_callback: Option<std::sync::Arc<dyn Fn(&str) + Send + Sync>>,

    // ---------------------------------------------------------------
    // CUDA-only orchestration state.
    // ---------------------------------------------------------------
    #[cfg(feature = "cuda")]
    device: Option<GpuDevice>,
    #[cfg(feature = "cuda")]
    model: Option<CudaModel>,
    /// CUDA graph runner for decode batches (monolithic mode).
    #[cfg(feature = "cuda")]
    graph_runner: Option<CudaGraphRunner>,
    /// CUDA graph runner for single-sequence prefill batches.
    #[cfg(feature = "cuda")]
    prefill_graph_runner: Option<PrefillGraphRunner>,
    #[cfg(feature = "cuda")]
    last_graph_batch_size: Option<usize>,
    #[cfg(feature = "cuda")]
    graph_metadata_valid: bool,
    /// True when the model uses GGML quantized layers (disables CUDA graphs).
    #[cfg(feature = "cuda")]
    uses_ggml: bool,
    /// Pre-allocated pinned host staging buffers for graph replay.
    #[cfg(feature = "cuda")]
    host_staging: Option<HostStaging>,
    /// Set once per thread to avoid redundant `ctx_set_current` driver calls.
    #[cfg(feature = "cuda")]
    ctx_set_on_thread: bool,
    /// Deferred D2H commit from the previous greedy graph step.
    #[cfg(feature = "cuda")]
    pending_commit: Option<PendingCommit>,
    /// Per-request grammar guide state for constrained decoding.
    #[cfg(all(feature = "cuda", feature = "guided-decoding"))]
    grammar_states: HashMap<String, vllm_model::grammar::GrammarGuide>,
    /// Parser factory for grammar-guided decoding (built once from tokenizer).
    #[cfg(all(feature = "cuda", feature = "guided-decoding"))]
    grammar_factory: Option<std::sync::Arc<vllm_model::grammar::LlgParserFactory>>,
    /// LogitsProcessor pipeline: persistent GPU state, rebuilt only on batch changes.
    #[cfg(feature = "cuda")]
    logits_pipeline: Option<LogitsProcessorPipeline>,
    /// Grammar mask processor (separate from pipeline — needs backup logits).
    #[cfg(feature = "cuda")]
    grammar_processor: GrammarMaskProcessor,
    /// Allowed token IDs processor.
    #[cfg(feature = "cuda")]
    allowed_token_ids_processor: AllowedTokenIdsProcessor,
    /// Seal-pad processor: forces pad tokens after EOS for sealed requests.
    #[cfg(feature = "cuda")]
    seal_pad_processor: SealPadProcessor,
    /// True when KV cache uses FP8 E4M3 quantization.
    #[cfg(feature = "cuda")]
    kv_cache_is_fp8: bool,
    #[cfg(feature = "cuda")]
    _calculate_kv_scales: bool,
    #[cfg(feature = "cuda")]
    _k_scale_constant: f32,
    #[cfg(feature = "cuda")]
    _v_scale_constant: f32,
    /// GPU weight allocations tracked for sleep/wake lifecycle.
    #[cfg(feature = "cuda")]
    weight_gpu_allocs: Vec<vllm_cuda::RawGpuMem>,
    /// Saved num_gpu_blocks for re-init after wake.
    #[cfg(feature = "cuda")]
    num_gpu_blocks_saved: usize,
    // Pipeline-parallel (PP) plumbing was removed alongside the hand-written
    // CUDA model forwards: the only `forward_pp` impls lived on
    // `vllm_cuda::model::{llama,qwen2,gemma2,gemma3}` and the send/recv
    // intermediates path was wired straight into those types. Ferrite-forward
    // does not yet provide a PP-aware forward; load_model fails fast when
    // `pp_size > 1` rather than silently routing one rank into a stub.

    // ---------------------------------------------------------------
    // Metal-only state (Phase F: cfg-mutex extension).
    // ---------------------------------------------------------------
    /// Shared `MetalDevice` handle. Set in `init_device`; carries
    /// `recommended_max_working_set_size` / `current_allocated_size`
    /// for `determine_available_memory`'s pre-load profile path.
    #[cfg(feature = "metal")]
    metal_device: Option<std::sync::Arc<ferrite_metal_kernels::device::MetalDevice>>,
    /// Device + command queue + caching allocator. Built lazily in
    /// `load_model` from `metal_device.device.clone()` once we know we
    /// will be loading weights through ferrite. Holds the `GpuWeights`
    /// allocator (via `Arc<MetalAllocator>` shared with the per-arch
    /// `Weights::load` upload path) plus the per-step CommandQueue.
    #[cfg(feature = "metal")]
    gpu_device: Option<GpuDevice>,
    /// Loaded ferrite-forward weights — `Box<dyn FerriteWeights>`
    /// dispatched through `try_load`. The trait `forward` body
    /// collapses to the per-canonical metal `forward` fn under
    /// cfg(metal); the worker calls it through the trait vtable.
    #[cfg(feature = "metal")]
    model: Option<Box<dyn ferrite_forward::FerriteWeights>>,
    /// Draft model for speculative decoding (loaded onto the same
    /// device + allocator + command queue as the target). Idle until
    /// `DRAFT_SPEC_DECODE_PLAN.md` phase 4 wires the proposer.
    #[cfg(feature = "metal")]
    draft_model: Option<Box<dyn ferrite_forward::FerriteWeights>>,
    /// Compiled greedy-sampling pipeline. Cached once at load_model
    /// to avoid recompiling the MSL kernel each step.
    #[cfg(feature = "metal")]
    argmax_kernels: Option<ferrite_metal_kernels::argmax::ArgmaxKernels>,
    /// Phase 6 chain-advance kernel. One small kernel that bumps
    /// per-req `runtime.positions` / `slot_mapping` / `seqused_k` in
    /// place between K-step chain iters. Cached at load_model so the
    /// pipeline is built once.
    #[cfg(feature = "metal")]
    chain_advance_kernel: Option<ferrite_metal_kernels::chain_advance::ChainAdvanceKernel>,
    /// Second `MTLCommandQueue` on the same device, dedicated to the
    /// draft chain. Metal device-level parallelism: dispatches on
    /// distinct queues run concurrently on Apple Silicon when they
    /// don't contend for the same residency set / arena slots.
    ///
    /// Used by Phase 8: lockstep prefill submitted here in parallel
    /// with target verify on the main queue. Lockstep's draft-KV
    /// writes at target positions don't overlap with target's
    /// target-KV writes (separate KV pools), so they're safe to run
    /// concurrently. Allocated when the draft model loads.
    #[cfg(feature = "metal")]
    draft_queue: Option<
        ::objc2::rc::Retained<::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTLCommandQueue>>,
    >,
}

// Safety: FerriteWorker contains raw GPU pointers (via GpuDevice, model weights,
// KV cache, PP buffers, CUDA graphs) and raw pointer fields in OwnedTensor /
// RawGpuAlloc / RawGpuMem. All GPU resources are allocated on a single CUDA
// context and accessed exclusively from the worker thread. Send is required
// because the worker is created on the main thread and moved to its dedicated
// worker thread via the executor's spawn.
unsafe impl Send for FerriteWorker {}

/// Resolve model path: local dir, local GGUF file, or HF download.
///
/// Returns a `PathBuf` that is either:
/// - A directory containing safetensors + config.json (normal path)
/// - A `.gguf` file path (GGUF path — load_model detects this)
///
/// Resolve a model identifier to a local path. Single canonical
/// implementation shared by every Worker `load_model` body and by
/// vllm-serve's parallel-tokenizer hoist.
///
/// Three cases:
/// 1. Local `.gguf` file path → returns as-is.
/// 2. Local directory → returns as-is.
/// 3. HuggingFace Hub model ID:
///    - First check `hf_hub::Cache` (network-free on-disk lookup).
///      Skips the ~100ms TLS handshake + ETag probe the Api path runs
///      even for fully cached models. MLX doesn't pay this cost.
///    - On cache miss, fall through to the Api path: download
///      config.json + tokenizer + safetensors shards (parallel, up
///      to 8 concurrent), or auto-detect a `.gguf` for GGUF repos.
///
/// Used to live duplicated in `gpu_worker_base.rs`; consolidated
/// here so the Hub plumbing (parallel shard download with progress
/// bars, GGUF auto-detect, etc.) has one home.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn resolve_model_path(
    model_path: &str,
    hf_token: Option<&str>,
    gguf_file: Option<&str>,
) -> ExecutorResult<PathBuf> {
    let path = Path::new(model_path);

    // Local .gguf file.
    if path.is_file() && path.extension().is_some_and(|e| e == "gguf") {
        return Ok(path.to_path_buf());
    }

    // Local directory.
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }

    // Network-free local-cache fast path. `hf_hub::Cache` resolves
    // refs/<revision> → snapshots/<commit>/<file> on disk without
    // any HTTP. If `config.json` is already there, the model has
    // been pulled before; return its dir directly. The `Api` path
    // would otherwise burn ~100ms on a TLS handshake + ETag probe
    // for a model we already have locally.
    if gguf_file.is_none() {
        let cache_repo = hf_hub::Cache::from_env().model(model_path.to_string());
        if let Some(config_path) = cache_repo.get("config.json")
            && let Some(model_dir) = config_path.parent().map(|p| p.to_path_buf())
        {
            info!(
                "Using cached model: {} (HF cache, no network)",
                model_dir.display()
            );
            return Ok(model_dir);
        }
    }

    info!("Downloading model from HuggingFace Hub: {model_path}");
    let mut builder = hf_hub::api::sync::ApiBuilder::from_env();
    if let Some(token) = hf_token {
        builder = builder.with_token(Some(token.to_string()));
    }
    let api = builder
        .build()
        .map_err(|e| ExecutorError::WorkerInit(format!("failed to build HF API: {e}")))?;
    let repo = api.model(model_path.to_string());

    // GGUF download: explicit filename or auto-detect from repo.
    let gguf_filename = gguf_file.map(String::from).or_else(|| {
        // Auto-detect: if model name looks like a GGUF repo, find smallest Q4_K_M file.
        if !model_path.to_ascii_uppercase().contains("GGUF") {
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
        for pattern in &["Q4_K_M", "Q4_K_S", "Q4_K", "Q4_0", "Q8_0"] {
            if let Some(f) = gguf_files.iter().find(|s| s.rfilename.contains(pattern)) {
                return Some(f.rfilename.clone());
            }
        }
        gguf_files.sort_by(|a, b| a.rfilename.cmp(&b.rfilename));
        Some(gguf_files[0].rfilename.clone())
    });
    if let Some(ref gguf_file) = gguf_filename {
        info!("Downloading GGUF file: {gguf_file}");
        let gguf_path = repo.get(gguf_file).map_err(|e| {
            ExecutorError::WorkerInit(format!("failed to download GGUF {gguf_file}: {e}"))
        })?;
        let _ = repo.get("tokenizer.json");
        let _ = repo.get("tokenizer_config.json");
        return Ok(gguf_path);
    }

    let config_path = repo
        .get("config.json")
        .map_err(|e| ExecutorError::WorkerInit(format!("failed to download config.json: {e}")))?;
    let model_dir = config_path.parent().unwrap().to_path_buf();

    let _ = repo.get("tokenizer.json");
    let _ = repo.get("tokenizer_config.json");

    if repo.get("model.safetensors").is_ok() {
        return Ok(model_dir);
    }
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let index = vllm_model::weight::SafeTensorsIndex::from_file(&index_path)
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to parse index: {e}")))?;
        let sorted_shards = index.shard_files();
        let total = sorted_shards.len();

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
                                repo.download_with_progress(shard, bar)
                                    .map(|_| ())
                                    .map_err(|e| {
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
        }
        return Ok(model_dir);
    }

    Err(ExecutorError::WorkerInit(format!(
        "no safetensors weights found for {model_path}"
    )))
}

#[cfg(feature = "cuda")]
impl FerriteWorker {
    pub fn new(config: FerriteWorkerConfig) -> Self {
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
            draft_model_dir: None,
            draft_hf_config: None,
            draft_kv_cache: None,
            model_dtype: GpuDType::BF16,
            resolved_architecture: None,
            is_shutdown: false,
            graph_runner: None,
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
            kv_cache_is_fp8,
            _calculate_kv_scales: calculate_kv_scales,
            _k_scale_constant: k_scale_constant,
            _v_scale_constant: v_scale_constant,
            weight_gpu_allocs: Vec::new(),
            num_gpu_blocks_saved: 0,
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

    /// PP plumbing was removed alongside the hand-written CUDA model
    /// forwards. These no-op stubs keep `vllm-serve`'s init code compiling
    /// while load_model rejects `pp_size > 1` so the calls are unreachable.
    #[cfg(feature = "nccl")]
    pub fn set_pp_group(&mut self, _group: std::sync::Arc<vllm_cuda::nccl::NcclGroup>) {}

    pub fn allocate_pp_recv_buffers(&mut self) {}

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
            info!("FerriteWorker: no tokenizer.json found, grammar-guided decoding unavailable");
            return;
        }
        let tokenizer_bytes = match std::fs::read(&tokenizer_path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    "FerriteWorker: failed to read tokenizer.json for grammar factory: {e}"
                );
                return;
            }
        };

        match vllm_model::grammar::build_parser_factory(&tokenizer_bytes) {
            Ok(factory) => {
                info!("FerriteWorker: grammar parser factory built");
                self.grammar_factory = Some(factory);
            }
            Err(e) => {
                tracing::warn!("FerriteWorker: failed to build grammar parser factory: {e}");
            }
        }
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

            let rejection = ::vllm_engine::spec_decode::greedy_rejection_sample(
                target_ids,
                &req_slice.spec_token_ids,
            );
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
            draft_seed_inputs: None,
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
}

// ---------------------------------------------------------------------------
// Worker trait implementation
// ---------------------------------------------------------------------------

// Phase F note: the cuda and metal `impl Worker for FerriteWorker` blocks
// are cfg-mutex'd — each compile sees exactly one. Method bodies that
// genuinely differ (load_model dispatcher, initialize_cache shape,
// execute_model body, sampling, determine_available_memory profile path,
// compile_or_warm_up_model) live separately rather than as `#[cfg]` arms
// inside one `fn`, because the cuda bodies are large and call cuda-only
// helpers that don't compile under metal. Per-method `#[cfg]` arms would
// expand to ~2k lines of dead-under-metal code that still has to type-
// check; cfg-gated blocks are the same shape with less surface.

// Phase 5.2a stub: CUDA gets the `SpecDecodeBackend` trait surface so
// downstream architecture can hold the worker behind `dyn
// SpecDecodeBackend` from day one. Bodies stay `NotImplemented` —
// phase 5.2b ports them (cudaMalloc + cuda forward +
// `vllm_cuda::kernels::argmax_batched`, lifting the existing
// `spec_decode_greedy_sample` logic).
#[cfg(feature = "cuda")]
impl ::vllm_engine::spec_decode::SpecDecodeBackend for FerriteWorker {
    fn forward_argmax_blocking(
        &mut self,
        _model: ::vllm_engine::spec_decode::ModelHandle,
        _kv_pool: ::vllm_engine::spec_decode::KvPoolHandle,
        _req: &::vllm_engine::spec_decode::ForwardArgmaxRequest<'_>,
    ) -> Result<Vec<u32>, ::vllm_engine::spec_decode::BackendError> {
        Err(::vllm_engine::spec_decode::BackendError::NotImplemented(
            "cuda: forward_argmax_blocking (phase 5.2b)",
        ))
    }

    fn load_secondary_model(
        &mut self,
        _path: &::std::path::Path,
        _dtype: Option<&str>,
    ) -> Result<::vllm_engine::spec_decode::ModelHandle, ::vllm_engine::spec_decode::BackendError>
    {
        Err(::vllm_engine::spec_decode::BackendError::NotImplemented(
            "cuda: load_secondary_model (phase 5.2b)",
        ))
    }

    fn allocate_kv_pool(
        &mut self,
        _model: ::vllm_engine::spec_decode::ModelHandle,
        _num_blocks: usize,
    ) -> Result<::vllm_engine::spec_decode::KvPoolHandle, ::vllm_engine::spec_decode::BackendError>
    {
        Err(::vllm_engine::spec_decode::BackendError::NotImplemented(
            "cuda: allocate_kv_pool (phase 5.2b)",
        ))
    }

    fn kv_per_block_bytes(
        &self,
        _model: ::vllm_engine::spec_decode::ModelHandle,
    ) -> Result<usize, ::vllm_engine::spec_decode::BackendError> {
        Err(::vllm_engine::spec_decode::BackendError::NotImplemented(
            "cuda: kv_per_block_bytes (phase 5.2b)",
        ))
    }
}

#[cfg(feature = "cuda")]
impl Worker for FerriteWorker {
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
        let model_dir = resolve_model_path(
            &self.config.model_path,
            self.config.hf_token.as_deref(),
            self.config.gguf_file.as_deref(),
        )?;
        info!("FerriteWorker: loading model from {}", model_dir.display());

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
        info!("FerriteWorker: using dtype {:?}", dtype);

        // 4. Look up architecture.
        let arch = hf_config.architectures.first().cloned().unwrap_or_default();
        info!("FerriteWorker: architecture = {arch}");

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
        info!("FerriteWorker: parsed {} weight tensors", weights.len());
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
            info!("FerriteWorker: merged {merged} LoRA weight tensors");
        }

        // 6. Detect quantization config.
        let qconfig = quant::detect_quant_config(&model_dir)
            .map_err(|e| ExecutorError::WorkerInit(format!("quant config detection: {e}")))?;
        if qconfig.is_quantized() {
            info!("FerriteWorker: detected quantization: {:?}", qconfig);
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

        // 7. Construct model. Ferrite-forward is the sole model-construction
        //    path now; hand-written `vllm_cuda::model::*` loaders + their
        //    PP/Qwen3Next-specific plumbing are gone. Pipeline parallelism is
        //    not supported until ferrite-forward grows native PP — fail fast
        //    rather than silently routing one rank into a stub.
        let tp_world = self.config.tp_world_size;
        let tp_rank = self.config.tp_rank;
        if self.config.pp_size > 1 {
            return Err(ExecutorError::WorkerInit(
                "pipeline parallelism is not supported on ferrite-only forwards \
                 (the previous PP path lived in `vllm_cuda::model::*` which \
                 has been removed). Run with `--pipeline-parallel-size 1`."
                    .into(),
            ));
        }

        // ── Hand off to ferrite-forward (all supported arches in one shot).
        //
        // `ferrite_forward::try_load` walks every `#[forward]`-
        // registered arch; first whose `hf_arches` list contains
        // `arch.as_str()` wins. Auto-registered via
        // `inventory::submit!` at macro expansion — adding a new
        // arch to ferrite-models touches zero lines here.
        let ferrite_loaded: Option<CudaModel> = {
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
                    "FerriteWorker: loaded {} via ferrite-forward ({})",
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
                        "FerriteWorker: loaded {} vision encoder via ferrite-forward",
                        arch
                    );
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
        };
        let model = ferrite_loaded.ok_or_else(|| {
            ExecutorError::WorkerInit(format!(
                "no ferrite-forward variant matched arch=`{arch}` at tp={tp_world}. \
                 Hand-written CUDA model forwards have been removed; ferrite-forward \
                 is the sole model-load path. Rebuild with the matching arch in \
                 `FERRITE_MODELS` (or unset to compile every variant), and confirm \
                 the arch's `configs/quantizations.json` lists this checkpoint's \
                 quantization scheme."
            ))
        })?;

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

        // Pre-allocate the FA3 scheduler-metadata buffer on Hopper
        // (sm_90+) before `determine_available_memory` runs its profile.
        // The 4 KB allocation lands as non-torch persistent memory and
        // is correctly accounted for in the KV-cache budget. On non-
        // Hopper devices the buffer is never read, so don't allocate.
        // FERRITE_DISABLE_FA3=1 also skips this — keeps the budget
        // identical between FA3-on and FA3-off A/B comparisons.
        #[cfg(fa3_built)]
        if let Some(ref dev) = self.device
            && dev.sm_version >= 90
            && std::env::var("FERRITE_DISABLE_FA3").ok().as_deref() != Some("1")
        {
            unsafe {
                ferrite_kernels::flash_attn_3::fa3_init_metadata();
            }
        }

        info!(
            "FerriteWorker: model loaded in {:.1}s",
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
                |bytes| {
                    let ptr = driver::mem_alloc(bytes)?;
                    Ok(vllm_cuda::RawGpuMem::new(ptr, bytes))
                },
            )
        }
        .map_err(|e| ExecutorError::WorkerInit(format!("KvCachePool: {e}")))?;

        self.kv_cache = Some(pool);
        // Qwen3Next GDN state pool was allocated here when the hand-written
        // `vllm_cuda::model::qwen3_next` lived. With ferrite-only forwards,
        // recurrent state is owned by ferrite-forward's per-arch glue.

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
        // - MoE: fused path uses ~55 MB/layer (OK), but profiling with
        //   max_num_batched_tokens can still OOM on the attention side
        // - Encoders: pooling-only, no logit head to drive the profile
        let is_moe = self.model.as_ref().is_some_and(|m| m.is_moe());
        let is_encoder = self.model.as_ref().is_some_and(|m| m.is_ferrite_encoder());
        if self.uses_ggml || is_moe || is_encoder {
            let tag = if self.uses_ggml {
                "GGML"
            } else if is_moe {
                "MoE"
            } else {
                "encoder"
            };
            info!(
                "FerriteWorker: {tag} model — skipping activation profiling, using fixed estimate"
            );
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
            "FerriteWorker: {:.0} MB free VRAM",
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
                |bytes| {
                    let ptr = driver::mem_alloc(bytes)?;
                    Ok(vllm_cuda::RawGpuMem::new(ptr, bytes))
                },
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
            info!("FerriteWorker: --enforce-eager set, skipping CUDA graph capture");
            return Ok(());
        }

        if self.uses_ggml {
            info!(
                "FerriteWorker: GGML model — skipping CUDA graph capture (incompatible with graph capture)"
            );
            return Ok(());
        }

        if self.kv_cache_is_fp8 {
            info!(
                "FerriteWorker: FP8 KV cache — skipping CUDA graph capture (variable scratch buffer sizes)"
            );
            return Ok(());
        }

        let max_blocks_per_seq = self.max_blocks_per_seq();

        let (model, kv_cache, device) =
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
                "FerriteWorker: resolved cuda_graph_mode {:?} → {:?} (SM{}, TP={})",
                self.config.cuda_graph_mode, resolved, device.sm_version, self.config.tp_world_size
            );
            self.config.cuda_graph_mode = resolved;
        }

        // Encoder models don't support CUDA graph capture (no decode loop).
        if model.is_ferrite_encoder() {
            self.config.cuda_graph_mode = CudaGraphMode::None;
            info!("FerriteWorker: encoder model — disabling CUDA graphs");
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

        // Piecewise CUDA-graph capture was removed alongside the hand-written
        // CUDA model forwards (the per-piece `execute_*_piece` bodies dispatched
        // through `vllm_cuda::model::*` types directly). Treat
        // `Piecewise` / `FullAndPiecewise` modes as monolithic-only requests;
        // the scheduling-mode `resolve()` already maps them to safe defaults.

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

        info!("FerriteWorker: sleeping (level {level}) — freeing GPU memory");

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

        info!("FerriteWorker: sleep complete — GPU memory released");
        Ok(())
    }

    fn wake_up(&mut self, _tags: Option<&[String]>) -> ExecutorResult<()> {
        info!("FerriteWorker: waking up — reloading model and KV cache");

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

        info!("FerriteWorker: wake complete");
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
} // end impl Worker for FerriteWorker

#[cfg(feature = "cuda")]
impl FerriteWorker {
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
        // PP plumbing was removed alongside the hand-written CUDA model
        // forwards; load_model rejects pp_size > 1, so this path is decoder-
        // only and always pp_active == false.

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
        let graph_bs = if is_decode && !any_mm_in_batch {
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
        let decode_graph_bs = if is_mixed && !any_spec_in_batch {
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

        // Piecewise CUDA-graph dispatch and the Qwen3Next-specific GDN
        // state-pool lookup were removed alongside the hand-written CUDA model
        // forwards; everything below uses monolithic graph or eager.

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

                {
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
                    // MRoPE override is per-arch — Qwen2-VL family
                    // packs (T, H, W) per token so MM tokens carry grid
                    // coords; Gemma3-MM / LLaVA-class use standard 1D
                    // RoPE in the text decoder and must keep the
                    // prebuilt sequence positions. The flag rides on
                    // `MultimodalForward::mm_metadata().mrope_positions`,
                    // declared per-arch on `pub const PROCESSOR:
                    // ferrite_vision::MmMetadata`.
                    let mrope = if let CudaModel::Ferrite(fm) = model {
                        fm.mm
                            .as_ref()
                            .map(|m| m.mm_metadata().mrope_positions)
                            .unwrap_or(false)
                    } else {
                        false
                    };
                    let gpu_positions_2d = if mrope && !per_req_mm.is_empty() {
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
} // end impl FerriteWorker (execute_model_inner)

/// Read-only batch context passed to `update_logits_processors`.
#[cfg(feature = "cuda")]
struct LogitsUpdateCtx<'a> {
    batch_changed: bool,
    num_reqs: usize,
    sampling_params_map: &'a HashMap<String, SamplingParams>,
    token_buffers: &'a HashMap<String, Vec<u32>>,
    batch_req_ids: &'a [String],
}

/// Per-block KV bytes for one model: layers × 2 (K+V) × heads × head_dim × block_size × 2 bytes.
/// bf16 and f16 are both 2 bytes/elt under metal; int4 KV is unsupported.
#[cfg(feature = "metal")]
fn kv_per_block_bytes(model: &dyn ferrite_forward::FerriteWeights, block_size: usize) -> usize {
    let elt_bytes: usize = 2;
    (model.num_hidden_layers() as usize)
        .saturating_mul(2)
        .saturating_mul(model.num_key_value_heads() as usize)
        .saturating_mul(model.head_dim() as usize)
        .saturating_mul(block_size)
        .saturating_mul(elt_bytes)
}

/// Compute KV cache budget matching Python vLLM's formula exactly:
///   requested = total_memory * gpu_memory_utilization
///   non_kv_cache = weights_and_overhead + peak_activations + 150 MiB
///   available_kv_bytes = requested - non_kv_cache
///
/// This is the exact logic used in `FerriteWorker::determine_available_memory`.
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
    /// `compute_available_kv_bytes` (the same formula FerriteWorker uses)
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

// ---------------------------------------------------------------------------
// Metal arm (Phase F: cfg-mutex extension; Step 2 = instantiate-only).
// ---------------------------------------------------------------------------
//
// The metal `new()` constructor + `Worker` impl live in their own
// cfg-gated blocks rather than as `#[cfg]` arms inside the cuda
// constructor / cuda Worker impl. The cuda body is ~3500 lines that
// call cuda-only helpers (CudaModel dispatch, CudaGraphRunner, NCCL
// PP, FP8 scales, LogitsProcessorPipeline, GdnStatePool, …); none
// of those compile under metal, so wrapping each cuda body with an
// `#[cfg(feature = "cuda")] { … }` arm would still drag the
// dead-under-metal code through type-checking. Per the Phase F
// rules, the FerriteWorker struct stays one struct with cfg-mutex'd
// fields; the trait impl uses cfg-mutex'd blocks for the same reason
// the macro emission uses cfg-mutex'd `Weights` blocks.
//
// Step 2 wires `init_device` / `shutdown` / `rank` etc. and stubs the
// heavy lifecycle methods (`load_model`, `initialize_cache`,
// `execute_model`, `compile_or_warm_up_model`,
// `determine_available_memory`) so vllm-serve can instantiate a
// FerriteWorker under metal. Step 3 fills those bodies in via the
// per-arch `metal_pool()` factory + `MetalWorkerPool::forward`.

#[cfg(feature = "metal")]
impl FerriteWorker {
    pub fn new(config: FerriteWorkerConfig) -> Self {
        Self {
            config,
            kv_cache: None,
            model_dir: None,
            hf_config: None,
            // Metal forward currently produces f16 outputs; cuda's "auto"
            // resolves to bf16 from the checkpoint, but ferrite-metal's
            // shaders are specialized on f16 and the worker only needs
            // this dtype to size KV cache slots, which Step 3 will revise.
            model_dtype: GpuDType::F16,
            resolved_architecture: None,
            is_shutdown: false,
            token_buffers: HashMap::new(),
            prompt_lengths: HashMap::new(),
            annotation_buffers: HashMap::new(),
            mm_data_buffers: HashMap::new(),
            sampling_params_map: HashMap::new(),
            input_batch: InputBatch::new(),
            preloaded_tokenizer: None,
            seeded_rngs: HashMap::new(),
            progress_callback: None,
            metal_device: None,
            gpu_device: None,
            model: None,
            draft_model: None,
            draft_model_dir: None,
            draft_hf_config: None,
            draft_kv_cache: None,
            argmax_kernels: None,
            chain_advance_kernel: None,
            draft_queue: None,
        }
    }

    /// Set the progress callback for startup loading.
    pub fn set_progress_callback(&mut self, cb: std::sync::Arc<dyn Fn(&str) + Send + Sync>) {
        self.progress_callback = Some(cb);
    }

    /// Expose the HF config after load_model. Mirrors the cuda-side
    /// accessor — vllm-serve reads this to drive `compute_num_blocks`
    /// and the cache-init path.
    pub fn hf_config(&self) -> Option<&HfModelConfig> {
        self.hf_config.as_ref()
    }

    /// Expose the model directory after load_model.
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }

    /// Bytes per element for the model's KV cache dtype. Metal forces
    /// f16 for both compute and cache, so this is a constant; we keep
    /// the same accessor name as the cuda side so vllm-serve treats
    /// the worker uniformly.
    pub fn resolved_dtype_elem_bytes(&self) -> usize {
        self.model_dtype.size_bytes()
    }
}

// Inherent impl block for metal-only helpers that aren't on the
// `Worker` trait — kept separate from the trait impl below so non-
// trait methods don't get implicitly added to `Worker`.
#[cfg(feature = "metal")]
impl FerriteWorker {
    /// Load the speculative draft model onto the same Metal device +
    /// allocator + command queue as the target. Called from
    /// `load_model` after the target is fully loaded.
    ///
    /// Skips the metal device setup (reuses `self.gpu_device`), the
    /// argmax kernel compile (per-device, not per-model — `self.argmax_*`
    /// stays target-aligned for batch sizing), and the worker-pool /
    /// SpecializedPipelineCache for the draft is created lazily inside
    /// `ferrite_forward::try_load`. Two FerriteWeights instances on the
    /// same device → two pipeline caches; that's fine because cache
    /// keys include the library name and each model's synthesized
    /// libraries are derived from its own per-arch macro expansion.
    fn load_draft_model_metal(&mut self) -> ExecutorResult<()> {
        let path = self.config.draft_model_path.as_deref().ok_or_else(|| {
            ExecutorError::WorkerInit(
                "load_draft_model_metal called without draft_model_path".into(),
            )
        })?;

        let t_resolve = std::time::Instant::now();
        let draft_dir = resolve_model_path(path, self.config.hf_token.as_deref(), None)?;
        info!(
            "FerriteWorker(metal): resolved draft model dir in {:?} ({})",
            t_resolve.elapsed(),
            draft_dir.display()
        );

        if draft_dir.is_file() && draft_dir.extension().is_some_and(|e| e == "gguf") {
            return Err(ExecutorError::WorkerInit(
                "draft GGUF not supported on metal (safetensors only)".into(),
            ));
        }

        let draft_hf_config = HfModelConfig::from_path(&draft_dir)
            .map_err(|e| ExecutorError::WorkerInit(format!("draft config parse failed: {e}")))?;
        let draft_arch = draft_hf_config
            .architectures
            .first()
            .cloned()
            .unwrap_or_default();

        // Reuse the target's gpu_device — its allocator owns the
        // residency set the target weights live in, and the draft must
        // share that set or attention reads will race the Apple pager.
        // `MetalAllocator::clone` shares the underlying arena via
        // Arc<Mutex<…>>, so a weight allocated through the draft clone
        // is reachable via the target's clone too.
        let gpu_device = self.gpu_device.as_ref().ok_or_else(|| {
            ExecutorError::WorkerInit(
                "gpu_device not initialized — draft load must run after target load".into(),
            )
        })?;
        let allocator = (*gpu_device.allocator).clone();

        let t_from_dir = std::time::Instant::now();
        let mut draft_weights = GpuWeights::from_dir(&draft_dir, allocator)
            .map_err(|e| ExecutorError::WorkerInit(format!("draft weight load failed: {e}")))?;
        info!(
            "FerriteWorker(metal): draft GpuWeights::from_dir in {:?} ({} tensors)",
            t_from_dir.elapsed(),
            draft_weights.len()
        );
        draft_weights.set_target_dtype(GpuDType::BF16);

        let hf_fp = ferrite_forward::HfFingerprint {
            rope_scaling_type: draft_hf_config
                .extra
                .get("rope_scaling")
                .and_then(|rs| rs.get("rope_type").or_else(|| rs.get("type")))
                .and_then(|v| v.as_str()),
            rope_scaling_hash: draft_hf_config
                .extra
                .get("rope_scaling")
                .map(ferrite_forward::hash_json_value),
        };
        let max_model_len = self
            .config
            .max_model_len
            .or(draft_hf_config.max_position_embeddings)
            .unwrap_or(4096);

        let t_try_load = std::time::Instant::now();
        let draft_model = ferrite_forward::try_load(
            &mut draft_weights,
            (),
            draft_arch.as_str(),
            1,
            0,
            max_model_len,
            hf_fp,
        )
        .map_err(|e| ExecutorError::WorkerInit(format!("draft try_load: {e}")))?
        .ok_or_else(|| ExecutorError::ArchNotSupported(draft_arch.clone()))?;

        info!(
            "FerriteWorker(metal): draft try_load in {:?} ({} via ferrite-forward, {})",
            t_try_load.elapsed(),
            draft_arch,
            draft_model.arch_name()
        );

        self.draft_model = Some(draft_model);
        self.draft_model_dir = Some(draft_dir);
        self.draft_hf_config = Some(draft_hf_config);

        // Phase 8 foundation: dedicated MTLCommandQueue for the draft
        // chain. Two queues on the same device run concurrently on
        // Apple Silicon; this enables lockstep prefill to overlap
        // with target verify (separate KV pools → no contention).
        if self.draft_queue.is_none() {
            let device = self
                .gpu_device
                .as_ref()
                .expect("gpu_device must be initialized before draft model loads");
            self.draft_queue = Some(
                device
                    .device
                    .newCommandQueue()
                    .expect("newCommandQueue for draft_queue returned nil"),
            );
            info!("FerriteWorker(metal): allocated dedicated draft MTLCommandQueue");
        }
        Ok(())
    }

    /// Allocate the draft model's KV pool. Mirrors `initialize_cache`'s
    /// target-pool path: StorageModePrivate buffers pinned into the
    /// allocator's shared residency set so attention reads don't race
    /// the Apple pager.
    fn initialize_draft_cache_metal(&mut self, num_gpu_blocks: usize) -> ExecutorResult<()> {
        let model = self.draft_model.as_ref().ok_or_else(|| {
            ExecutorError::WorkerInit(
                "initialize_draft_cache_metal called before draft_model load".into(),
            )
        })?;
        // Phase 4 of DRAFT_SPEC_DECODE_PLAN.md: lockstep proposer needs a
        // 1:1 mirror — every target block has a sibling draft block at the
        // same index. `determine_available_memory` already carved off
        // `draft / (target + draft)` of the KV budget for us, so the
        // engine's `num_gpu_blocks` is the post-split target count and
        // the draft can mirror it exactly. `FERRITE_DRAFT_KV_BLOCKS=<N>`
        // remains as an explicit-testing override.
        let draft_blocks = std::env::var("FERRITE_DRAFT_KV_BLOCKS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(num_gpu_blocks);
        let device = self
            .gpu_device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("gpu_device not initialized".into()))?;

        let cache_dtype = match model.metal_dtype() {
            ferrite_forward::interpreter::metal::MetalDtype::Bf16 => GpuDType::BF16,
            ferrite_forward::interpreter::metal::MetalDtype::F16 => GpuDType::F16,
            ferrite_forward::interpreter::metal::MetalDtype::Int4 => {
                return Err(ExecutorError::WorkerInit(
                    "draft KV cache cannot be int4-quantized".into(),
                ));
            }
        };

        let mtl_device = device.device.clone();
        let residency = device.allocator.residency().clone();

        let t_pool = std::time::Instant::now();
        let pool = unsafe {
            KvCachePool::new(
                model.num_hidden_layers() as usize,
                draft_blocks,
                self.config.block_size,
                model.num_key_value_heads() as usize,
                model.head_dim() as usize,
                cache_dtype,
                |bytes| {
                    let buffer = mtl_device
                        .newBufferWithLength_options(
                            bytes,
                            ::objc2_metal::MTLResourceOptions::StorageModePrivate,
                        )
                        .expect("newBufferWithLength_options returned nil");
                    residency.insert(&buffer);
                    Ok(RawGpuMem::from_buffer(buffer))
                },
            )
        }
        .map_err(|e| ExecutorError::WorkerInit(format!("draft KvCachePool: {e}")))?;
        info!(
            "FerriteWorker(metal): draft KV pool ready in {:?} ({} layers × {} blocks × {} tokens, \
             1:1 mirror of target's {} blocks)",
            t_pool.elapsed(),
            model.num_hidden_layers(),
            draft_blocks,
            self.config.block_size,
            num_gpu_blocks,
        );
        self.draft_kv_cache = Some(pool);
        Ok(())
    }

    /// Allocate a `StorageModeShared` u32 buffer and memcpy `data` into it.
    /// Shared between the verify path (via the `SpecDecodeBackend` trait
    /// impl) and the K-step draft chain inlined in `execute_model`.
    fn alloc_shared_u32_buf(
        device: &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTLDevice>,
        data: &[u32],
    ) -> ::objc2::rc::Retained<::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTLBuffer>> {
        let bytes = (data.len().max(1)) * 4;
        let buf = device
            .newBufferWithLength_options(
                bytes,
                ::objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithLength_options returned nil");
        if !data.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    buf.contents().as_ptr() as *mut u32,
                    data.len(),
                );
            }
        }
        buf
    }
}

/// Phase 6 K-step chain dispatch — the actual work of
/// [`FerriteWorker::forward_chain_k`] extracted into a free fn so the
/// Phase 9 speculative path (worker's lockstep thread, parallel to
/// target verify) can call it too without `&self` access. Caller
/// supplies the model + KV cache + GpuDevice + kernel refs.
///
/// Allocates K argmax_dual_write argument tables + per-iter argmax
/// output buffers + one stable chain_advance arg table + a packed
/// 16-byte consts buffer, then drives K forwards on one MTL4 CB via
/// `model.metal_chain_with_encoder`. After CB completion, host-reads
/// the K argmax buffers and returns iter-major `[k][num_reqs]`
/// argmax IDs.
#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
fn metal_chain_dispatch(
    model_ref: &dyn ferrite_forward::FerriteWeights,
    kv_cache_ref: &KvCachePool,
    device_mut: &mut GpuDevice,
    argmax_kernels: &ferrite_metal_kernels::argmax::ArgmaxKernels,
    chain_kernel: &ferrite_metal_kernels::chain_advance::ChainAdvanceKernel,
    req: &::vllm_engine::spec_decode::ForwardArgmaxRequest<'_>,
    block_size: usize,
    k: usize,
) -> Result<Vec<Vec<u32>>, String> {
    use ::objc2_metal::{MTL4ArgumentTable, MTLBuffer};

    if k == 0 {
        return Ok(Vec::new());
    }
    let num_reqs = req.cu_seqlens_q.len().saturating_sub(1);
    if num_reqs == 0 || num_reqs != req.num_tokens {
        return Err(format!(
            "metal_chain_dispatch: K-step decode requires num_tokens == num_reqs \
             (got num_tokens={}, num_reqs={})",
            req.num_tokens, num_reqs
        ));
    }

    let mtl_device = device_mut.device.clone();

    // ── 1. Upload iter-0 inputs ─────────────────────────────────
    let buf_input_ids = FerriteWorker::alloc_shared_u32_buf(&mtl_device, req.input_ids);
    let buf_positions = FerriteWorker::alloc_shared_u32_buf(&mtl_device, req.positions);
    let buf_slot_mapping = FerriteWorker::alloc_shared_u32_buf(&mtl_device, req.slot_mapping);
    let buf_cu_seqlens = FerriteWorker::alloc_shared_u32_buf(&mtl_device, req.cu_seqlens_q);
    let buf_seqused_k = FerriteWorker::alloc_shared_u32_buf(&mtl_device, req.seqused_k);
    let buf_block_table = FerriteWorker::alloc_shared_u32_buf(&mtl_device, req.block_table);

    let dtype_u32 = ferrite_cuda_core::dtype::DType::U32;
    let view_input_ids = unsafe {
        TensorView::from_raw(GpuTensor::new(
            buf_input_ids.contents().as_ptr() as *mut u8,
            &[req.num_tokens.max(1)],
            dtype_u32,
        ))
    };
    let view_positions = unsafe {
        TensorView::from_raw(GpuTensor::new(
            buf_positions.contents().as_ptr() as *mut u8,
            &[req.num_tokens.max(1)],
            dtype_u32,
        ))
    };
    let view_slot_mapping = unsafe {
        TensorView::from_raw(GpuTensor::new(
            buf_slot_mapping.contents().as_ptr() as *mut u8,
            &[req.num_tokens.max(1)],
            dtype_u32,
        ))
    };
    let view_cu_seqlens = unsafe {
        TensorView::from_raw(GpuTensor::new(
            buf_cu_seqlens.contents().as_ptr() as *mut u8,
            &[req.cu_seqlens_q.len().max(1)],
            dtype_u32,
        ))
    };
    let view_seqused_k = unsafe {
        TensorView::from_raw(GpuTensor::new(
            buf_seqused_k.contents().as_ptr() as *mut u8,
            &[req.seqused_k.len().max(1)],
            dtype_u32,
        ))
    };
    let view_block_table = unsafe {
        TensorView::from_raw(GpuTensor::new(
            buf_block_table.contents().as_ptr() as *mut u8,
            &[num_reqs.max(1), req.block_table_stride.max(1)],
            dtype_u32,
        ))
    };

    // ── 2. Allocate K argmax output buffers (host-visible) ──────
    let argmax_bytes = (req.num_tokens.max(1)) * 4;
    let argmax_bufs: Vec<
        ::objc2::rc::Retained<::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTLBuffer>>,
    > = (0..k)
        .map(|_| {
            mtl_device
                .newBufferWithLength_options(
                    argmax_bytes,
                    ::objc2_metal::MTLResourceOptions::StorageModeShared,
                )
                .expect("argmax_buf alloc returned nil")
        })
        .collect();

    // ── 3. Pack constants ───────────────────────────────────────
    let consts_buf = mtl_device
        .newBufferWithLength_options(16, ::objc2_metal::MTLResourceOptions::StorageModeShared)
        .expect("consts_buf alloc returned nil");
    let vocab_u32 = model_ref.vocab_size() as u32;
    unsafe {
        let p = consts_buf.contents().as_ptr() as *mut u32;
        *p.add(0) = req.num_tokens as u32;
        *p.add(1) = vocab_u32;
        *p.add(2) = block_size as u32;
        *p.add(3) = req.block_table_stride as u32;
    }

    // ── 4. Build K argmax_dual_write argument tables ────────────
    let argmax_arg_tables: Vec<_> = (0..k)
        .map(|_| {
            use ::objc2_metal::MTL4ArgumentTableDescriptor;
            let desc = MTL4ArgumentTableDescriptor::new();
            desc.setMaxBufferBindCount(5);
            mtl_device
                .newArgumentTableWithDescriptor_error(&desc)
                .expect("argmax dual_write arg_table alloc returned nil")
        })
        .collect();

    // ── 5. Build chain_advance argument table ───────────────────
    let chain_arg_table = {
        use ::objc2_metal::MTL4ArgumentTableDescriptor;
        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(7);
        mtl_device
            .newArgumentTableWithDescriptor_error(&desc)
            .expect("chain_advance arg_table alloc returned nil")
    };

    // ── 6. ForwardCtx ───────────────────────────────────────────
    let ctx = ferrite_forward::ForwardCtx {
        input_ids: view_input_ids,
        positions: view_positions,
        slot_mapping: view_slot_mapping,
        cu_seqlens_q: view_cu_seqlens,
        seqused_k: view_seqused_k,
        block_table: view_block_table,
        max_seqlen_q: req.max_seqlen_q,
        max_seqlen_k: req.max_seqlen_k,
        kv_cache: kv_cache_ref,
        has_spec_tokens: false,
    };

    // SAFETY: kernel refs survive the synchronous call below.
    let argmax_kernels_addr: usize =
        argmax_kernels as *const ferrite_metal_kernels::argmax::ArgmaxKernels as usize;
    let chain_kernel_addr: usize =
        chain_kernel as *const ferrite_metal_kernels::chain_advance::ChainAdvanceKernel as usize;
    let dtype = model_ref.metal_dtype();

    let argmax_bufs_for_closure = argmax_bufs.clone();
    let consts_buf_for_closure = consts_buf.clone();
    let argmax_arg_tables_for_closure = argmax_arg_tables.clone();
    let chain_arg_table_for_closure = chain_arg_table.clone();

    let num_tokens_u32 = req.num_tokens as u32;
    let num_reqs_u32 = num_reqs as u32;
    let k_usize = k;

    let body: ferrite_forward::MetalChainBody<'_> = Box::new(
        move |handle, runtime, enc| -> Result<(), String> {
            let argmax_kernels_ref: &ferrite_metal_kernels::argmax::ArgmaxKernels = unsafe {
                &*(argmax_kernels_addr as *const ferrite_metal_kernels::argmax::ArgmaxKernels)
            };
            let chain_kernel_ref: &ferrite_metal_kernels::chain_advance::ChainAdvanceKernel = unsafe {
                &*(chain_kernel_addr
                    as *const ferrite_metal_kernels::chain_advance::ChainAdvanceKernel)
            };

            let logits_buf = handle.logits_buf();
            let vocab = handle.vocab();
            let logits_addr = logits_buf.gpuAddress();
            let consts_addr = consts_buf_for_closure.gpuAddress();
            let next_in_addr = runtime.input_ids.gpuAddress();

            unsafe {
                chain_arg_table_for_closure.setAddress_atIndex(runtime.positions.gpuAddress(), 0);
                chain_arg_table_for_closure
                    .setAddress_atIndex(runtime.slot_mapping.gpuAddress(), 1);
                chain_arg_table_for_closure.setAddress_atIndex(runtime.seq_used_k.gpuAddress(), 2);
                chain_arg_table_for_closure.setAddress_atIndex(runtime.block_table.gpuAddress(), 3);
                chain_arg_table_for_closure.setAddress_atIndex(consts_addr + 8, 4);
                chain_arg_table_for_closure.setAddress_atIndex(consts_addr + 12, 5);
                chain_arg_table_for_closure.setAddress_atIndex(consts_addr, 6);
            }

            for iter in 0..k_usize {
                handle.run_forward_step(enc, num_tokens_u32, num_reqs_u32, false)?;

                let table = &argmax_arg_tables_for_closure[iter];
                let out_addr = argmax_bufs_for_closure[iter].gpuAddress();
                unsafe {
                    table.setAddress_atIndex(logits_addr, 0);
                    table.setAddress_atIndex(out_addr, 1);
                    table.setAddress_atIndex(consts_addr, 2);
                    table.setAddress_atIndex(consts_addr + 4, 3);
                    table.setAddress_atIndex(next_in_addr, 4);
                }
                let _ = vocab;
                match dtype {
                    ferrite_forward::interpreter::metal::MetalDtype::F16 => {
                        ferrite_metal_kernels::argmax::encode_argmax_f16_dual_write_into_mtl4(
                            argmax_kernels_ref,
                            enc,
                            table,
                            num_reqs_u32,
                        )
                        .map_err(|e| format!("encode argmax_f16_dual_write: {e:?}"))?;
                    }
                    ferrite_forward::interpreter::metal::MetalDtype::Bf16 => {
                        ferrite_metal_kernels::argmax::encode_argmax_bf16_dual_write_into_mtl4(
                            argmax_kernels_ref,
                            enc,
                            table,
                            num_reqs_u32,
                        )
                        .map_err(|e| format!("encode argmax_bf16_dual_write: {e:?}"))?;
                    }
                    ferrite_forward::interpreter::metal::MetalDtype::Int4 => {
                        return Err("argmax: int4 dtype has no direct kernel".into());
                    }
                }

                if iter + 1 < k_usize {
                    ferrite_metal_kernels::chain_advance::encode_chain_advance_into_mtl4(
                        chain_kernel_ref,
                        enc,
                        &chain_arg_table_for_closure,
                        num_reqs_u32,
                    )
                    .map_err(|e| format!("encode chain_advance: {e:?}"))?;
                }
            }
            Ok(())
        },
    );

    let prof = std::env::var_os("FERRITE_SPEC_PROFILE").is_some();
    let t_chain = std::time::Instant::now();
    unsafe { model_ref.metal_chain_with_encoder(&ctx, device_mut, req.num_tokens as u64, body)? };
    let chain_us = t_chain.elapsed().as_micros();

    let t_read = std::time::Instant::now();
    let mut out: Vec<Vec<u32>> = Vec::with_capacity(k);
    for buf in argmax_bufs.iter() {
        let slice: &[u32] = unsafe {
            std::slice::from_raw_parts(buf.contents().as_ptr() as *const u32, num_reqs.max(1))
        };
        out.push(slice[..num_reqs.max(1)].to_vec());
    }
    let read_us = t_read.elapsed().as_micros();

    if prof {
        eprintln!(
            "[spec-prof chain] k={} num_reqs={} chain={}us read={}us",
            k, num_reqs, chain_us, read_us,
        );
    }
    Ok(out)
}

#[cfg(feature = "metal")]
impl ::vllm_engine::spec_decode::SpecDecodeBackend for FerriteWorker {
    fn forward_argmax_blocking(
        &mut self,
        model: ::vllm_engine::spec_decode::ModelHandle,
        kv_pool: ::vllm_engine::spec_decode::KvPoolHandle,
        req: &::vllm_engine::spec_decode::ForwardArgmaxRequest<'_>,
    ) -> Result<Vec<u32>, ::vllm_engine::spec_decode::BackendError> {
        use ::vllm_engine::spec_decode::{BackendError, KvPoolHandle, ModelHandle};

        // Resolve handles. 0 → target, 1 → draft; everything else is
        // an unknown handle.
        let model_ref: &dyn ferrite_forward::FerriteWeights = match model {
            ModelHandle::TARGET => self
                .model
                .as_deref()
                .ok_or_else(|| BackendError::Backend("target model not loaded".into()))?,
            ModelHandle(1) => self
                .draft_model
                .as_deref()
                .ok_or_else(|| BackendError::Backend("draft model not loaded".into()))?,
            _ => return Err(BackendError::UnknownHandle("ModelHandle")),
        };
        let kv_cache_ref: &KvCachePool =
            match kv_pool {
                KvPoolHandle::TARGET => self.kv_cache.as_ref().ok_or_else(|| {
                    BackendError::Backend("target kv_cache not initialized".into())
                })?,
                KvPoolHandle(1) => self.draft_kv_cache.as_ref().ok_or_else(|| {
                    BackendError::Backend("draft kv_cache not initialized".into())
                })?,
                _ => return Err(BackendError::UnknownHandle("KvPoolHandle")),
            };

        let device_buf = self
            .gpu_device
            .as_ref()
            .ok_or_else(|| BackendError::Backend("gpu_device not initialized".into()))?;
        let mtl_device = device_buf.device.clone();
        let argmax_kernels = self
            .argmax_kernels
            .as_ref()
            .ok_or_else(|| BackendError::Backend("argmax_kernels not built".into()))?;

        // ── 1. Upload host slices to fresh shared-storage MTLBuffers ─────
        let prof = std::env::var_os("FERRITE_SPEC_PROFILE").is_some();
        let t_alloc = std::time::Instant::now();
        let buf_input_ids = Self::alloc_shared_u32_buf(&mtl_device, req.input_ids);
        let buf_positions = Self::alloc_shared_u32_buf(&mtl_device, req.positions);
        let buf_slot_mapping = Self::alloc_shared_u32_buf(&mtl_device, req.slot_mapping);
        let buf_cu_seqlens = Self::alloc_shared_u32_buf(&mtl_device, req.cu_seqlens_q);
        let buf_seqused_k = Self::alloc_shared_u32_buf(&mtl_device, req.seqused_k);
        let buf_block_table = Self::alloc_shared_u32_buf(&mtl_device, req.block_table);
        let alloc_us = t_alloc.elapsed().as_micros();

        // ── 2. Wrap MTLBuffers in TensorViews. Dtype is purely
        //       descriptive; the macro-emitted forward only reads
        //       `as_raw().raw_ptr()` and `numel()`.
        let dtype_u32 = ferrite_cuda_core::dtype::DType::U32;
        let num_reqs = req.cu_seqlens_q.len().saturating_sub(1);
        let view_input_ids = unsafe {
            TensorView::from_raw(GpuTensor::new(
                buf_input_ids.contents().as_ptr() as *mut u8,
                &[req.num_tokens.max(1)],
                dtype_u32,
            ))
        };
        let view_positions = unsafe {
            TensorView::from_raw(GpuTensor::new(
                buf_positions.contents().as_ptr() as *mut u8,
                &[req.num_tokens.max(1)],
                dtype_u32,
            ))
        };
        let view_slot_mapping = unsafe {
            TensorView::from_raw(GpuTensor::new(
                buf_slot_mapping.contents().as_ptr() as *mut u8,
                &[req.num_tokens.max(1)],
                dtype_u32,
            ))
        };
        let view_cu_seqlens = unsafe {
            TensorView::from_raw(GpuTensor::new(
                buf_cu_seqlens.contents().as_ptr() as *mut u8,
                &[req.cu_seqlens_q.len().max(1)],
                dtype_u32,
            ))
        };
        let view_seqused_k = unsafe {
            TensorView::from_raw(GpuTensor::new(
                buf_seqused_k.contents().as_ptr() as *mut u8,
                &[req.seqused_k.len().max(1)],
                dtype_u32,
            ))
        };
        let view_block_table = unsafe {
            TensorView::from_raw(GpuTensor::new(
                buf_block_table.contents().as_ptr() as *mut u8,
                &[num_reqs.max(1), req.block_table_stride.max(1)],
                dtype_u32,
            ))
        };

        // ── 3. Build ForwardCtx + run forward ────────────────────────────
        // Disjoint borrows: `model_ref`/`kv_cache_ref`/`argmax_kernels`
        // are immutable references into `self`; `device_mut` is a
        // mutable borrow of `self.gpu_device`. NLL accepts the split.
        //
        // Phase 8 routing: when the call is for the draft model
        // (ModelHandle(1)) and a dedicated `draft_queue` exists, we
        // build a shadow `GpuDevice` wrapping the same `MTLDevice` +
        // allocator but the draft queue, so the draft model's pool
        // attaches its residency set to that queue (instead of the
        // main queue). This lets target verify and draft work execute
        // concurrently on Apple Silicon. The shadow device is local
        // to this call; the worker's `gpu_device.queue` is untouched.
        let use_draft_queue = matches!(model, ModelHandle(1)) && self.draft_queue.is_some();
        let mut shadow_draft_device: Option<GpuDevice> = if use_draft_queue {
            let main_dev = self
                .gpu_device
                .as_ref()
                .ok_or_else(|| BackendError::Backend("gpu_device not initialized".into()))?;
            let draft_q = self.draft_queue.as_ref().unwrap().clone();
            Some(GpuDevice {
                device: main_dev.device.clone(),
                queue: draft_q,
                allocator: main_dev.allocator.clone(),
            })
        } else {
            None
        };
        let device_mut: &mut GpuDevice = if let Some(ref mut shadow) = shadow_draft_device {
            shadow
        } else {
            self.gpu_device
                .as_mut()
                .ok_or_else(|| BackendError::Backend("gpu_device not initialized".into()))?
        };

        let ctx = ferrite_forward::ForwardCtx {
            input_ids: view_input_ids,
            positions: view_positions,
            slot_mapping: view_slot_mapping,
            cu_seqlens_q: view_cu_seqlens,
            seqused_k: view_seqused_k,
            block_table: view_block_table,
            max_seqlen_q: req.max_seqlen_q,
            max_seqlen_k: req.max_seqlen_k,
            kv_cache: kv_cache_ref,
            has_spec_tokens: req.has_spec_tokens,
        };

        // ── 4. Fused argmax via forward_with_metal_followup. ─────────────
        // Phase 6a: argmax encodes onto the SAME MTL4 compute encoder
        // as the forward — forward + argmax share one CB, one commit,
        // one host wait. Pre-6a took the unfused path (separate dispatch
        // + commit + wait + readback) for 5.2a simplicity; this re-folds
        // it. Per call: 2 commit+waits → 1.
        let t_argbuf = std::time::Instant::now();
        let argmax_bytes = (req.num_tokens.max(1)) * 4;
        let argmax_out = mtl_device
            .newBufferWithLength_options(
                argmax_bytes,
                ::objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .expect("argmax_out alloc returned nil");
        let consts_buf = mtl_device
            .newBufferWithLength_options(8, ::objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("argmax consts alloc returned nil");
        let arg_table = {
            use ::objc2_metal::MTL4ArgumentTableDescriptor;
            let desc = MTL4ArgumentTableDescriptor::new();
            desc.setMaxBufferBindCount(4);
            mtl_device
                .newArgumentTableWithDescriptor_error(&desc)
                .expect("argmax arg_table alloc returned nil")
        };
        let argbuf_us = t_argbuf.elapsed().as_micros();
        let dtype = model_ref.metal_dtype();
        let argmax_out_for_closure = argmax_out.clone();
        let consts_for_closure = consts_buf.clone();
        let arg_table_for_closure = arg_table.clone();
        // SAFETY: `argmax_kernels` is borrowed from `&self.argmax_kernels`;
        // the closure runs synchronously inside `model_ref.forward_*` while
        // that borrow is live. Cast to a raw pointer to keep the closure
        // 'static + Send-safe per `MetalForwardFollowup`'s bound.
        let argmax_kernels_ptr: *const ferrite_metal_kernels::argmax::ArgmaxKernels =
            argmax_kernels;
        let argmax_kernels_addr = argmax_kernels_ptr as usize;
        let followup: ferrite_forward::MetalForwardFollowup<'_> = Box::new(
            move |enc, logits_buf, total_n_actual, vocab_actual| -> Result<(), String> {
                // Update consts buffer in-place (StorageModeShared).
                let consts_ptr = consts_for_closure.contents().as_ptr() as *mut u32;
                unsafe {
                    *consts_ptr = total_n_actual;
                    *consts_ptr.add(1) = vocab_actual;
                }
                use ::objc2_metal::{MTL4ArgumentTable, MTLBuffer};
                let logits_addr = logits_buf.gpuAddress();
                let out_addr = argmax_out_for_closure.gpuAddress();
                let consts_addr = consts_for_closure.gpuAddress();
                unsafe {
                    arg_table_for_closure.setAddress_atIndex(logits_addr, 0);
                    arg_table_for_closure.setAddress_atIndex(out_addr, 1);
                    arg_table_for_closure.setAddress_atIndex(consts_addr, 2);
                    arg_table_for_closure.setAddress_atIndex(consts_addr + 4, 3);
                }
                let argmax_kernels_ref: &ferrite_metal_kernels::argmax::ArgmaxKernels = unsafe {
                    &*(argmax_kernels_addr as *const ferrite_metal_kernels::argmax::ArgmaxKernels)
                };
                match dtype {
                    ferrite_forward::interpreter::metal::MetalDtype::F16 => {
                        ferrite_metal_kernels::argmax::encode_argmax_f16_into_mtl4(
                            argmax_kernels_ref,
                            enc,
                            &arg_table_for_closure,
                            total_n_actual,
                        )
                        .map_err(|e| format!("encode_argmax_f16: {e:?}"))?;
                    }
                    ferrite_forward::interpreter::metal::MetalDtype::Bf16 => {
                        ferrite_metal_kernels::argmax::encode_argmax_bf16_into_mtl4(
                            argmax_kernels_ref,
                            enc,
                            &arg_table_for_closure,
                            total_n_actual,
                        )
                        .map_err(|e| format!("encode_argmax_bf16: {e:?}"))?;
                    }
                    ferrite_forward::interpreter::metal::MetalDtype::Int4 => {
                        return Err("argmax: int4 dtype has no direct kernel".into());
                    }
                }
                Ok(())
            },
        );
        let t_fwd = std::time::Instant::now();
        let logits = unsafe {
            model_ref.forward_with_metal_followup(
                &ctx,
                device_mut,
                req.num_tokens as u64,
                Some(followup),
            )
        };
        let _ = logits; // argmax_out is what we read
        let fwd_us = t_fwd.elapsed().as_micros();

        // ── 5. Read host-visible argmax buffer + return. ─────────────────
        let t_read = std::time::Instant::now();
        let argmax_slice: &[u32] = unsafe {
            std::slice::from_raw_parts(
                argmax_out.contents().as_ptr() as *const u32,
                req.num_tokens.max(1),
            )
        };
        let out = argmax_slice[..req.num_tokens.max(1)].to_vec();
        let read_us = t_read.elapsed().as_micros();
        if prof {
            eprintln!(
                "[spec-prof] num_tokens={} num_seqs={} has_spec={} alloc={}us argbuf={}us fwd={}us read={}us",
                req.num_tokens, num_reqs, req.has_spec_tokens, alloc_us, argbuf_us, fwd_us, read_us,
            );
        }
        Ok(out)
    }

    /// Phase 6: K-step draft chain in ONE MTL4 command buffer. Forward,
    /// argmax_dual_write, and chain_advance (advances positions /
    /// slot_mapping / seqused_k in-place on the GPU between iters)
    /// for K iters share one CB, one commit, and one host wait. Per
    /// iter we spend ~kernel-time only — no host roundtrip, no per-
    /// iter allocator reset.
    ///
    /// Thin wrapper that resolves handles → refs and delegates to the
    /// free [`metal_chain_dispatch`] helper. The free fn is shared
    /// with the Phase 9 speculative path (worker's lockstep thread),
    /// which can't borrow `&self` while the main thread is mid-
    /// `forward_argmax_blocking` for target verify.
    fn forward_chain_k(
        &mut self,
        model: ::vllm_engine::spec_decode::ModelHandle,
        kv_pool: ::vllm_engine::spec_decode::KvPoolHandle,
        req: &::vllm_engine::spec_decode::ForwardArgmaxRequest<'_>,
        block_size: usize,
        k: usize,
    ) -> Result<Vec<Vec<u32>>, ::vllm_engine::spec_decode::BackendError> {
        use ::vllm_engine::spec_decode::{BackendError, KvPoolHandle, ModelHandle};

        if k == 0 {
            return Ok(Vec::new());
        }

        // Resolve target model + KV pool (handle 1 = draft, 0 = target).
        let model_ref: &dyn ferrite_forward::FerriteWeights = match model {
            ModelHandle::TARGET => self
                .model
                .as_deref()
                .ok_or_else(|| BackendError::Backend("target model not loaded".into()))?,
            ModelHandle(1) => self
                .draft_model
                .as_deref()
                .ok_or_else(|| BackendError::Backend("draft model not loaded".into()))?,
            _ => return Err(BackendError::UnknownHandle("ModelHandle")),
        };
        let kv_cache_ref: &KvCachePool =
            match kv_pool {
                KvPoolHandle::TARGET => self.kv_cache.as_ref().ok_or_else(|| {
                    BackendError::Backend("target kv_cache not initialized".into())
                })?,
                KvPoolHandle(1) => self.draft_kv_cache.as_ref().ok_or_else(|| {
                    BackendError::Backend("draft kv_cache not initialized".into())
                })?,
                _ => return Err(BackendError::UnknownHandle("KvPoolHandle")),
            };

        let argmax_kernels = self
            .argmax_kernels
            .as_ref()
            .ok_or_else(|| BackendError::Backend("argmax_kernels not built".into()))?;
        let chain_kernel = self
            .chain_advance_kernel
            .as_ref()
            .ok_or_else(|| BackendError::Backend("chain_advance_kernel not built".into()))?;

        // Phase 8 routing: when the call is for the draft model AND a
        // dedicated `draft_queue` exists, route through a shadow
        // GpuDevice on the draft queue so target verify and draft
        // chain stay disjoint.
        let use_draft_queue = matches!(model, ModelHandle(1)) && self.draft_queue.is_some();
        let mut shadow_draft_device: Option<GpuDevice> = if use_draft_queue {
            let main_dev = self.gpu_device.as_ref().expect("checked above");
            let draft_q = self.draft_queue.as_ref().unwrap().clone();
            Some(GpuDevice {
                device: main_dev.device.clone(),
                queue: draft_q,
                allocator: main_dev.allocator.clone(),
            })
        } else {
            None
        };
        let device_mut: &mut GpuDevice = if let Some(ref mut shadow) = shadow_draft_device {
            shadow
        } else {
            self.gpu_device
                .as_mut()
                .ok_or_else(|| BackendError::Backend("gpu_device not initialized".into()))?
        };

        metal_chain_dispatch(
            model_ref,
            kv_cache_ref,
            device_mut,
            argmax_kernels,
            chain_kernel,
            req,
            block_size,
            k,
        )
        .map_err(BackendError::Backend)
    }

    // (legacy body folded into the free fn `metal_chain_dispatch` below)

    fn load_secondary_model(
        &mut self,
        _path: &::std::path::Path,
        _dtype: Option<&str>,
    ) -> Result<::vllm_engine::spec_decode::ModelHandle, ::vllm_engine::spec_decode::BackendError>
    {
        // Thin wrapper: delegate to the existing helper which pulls
        // path + dtype from `self.config`. Phase 5.4 will tighten this
        // when `DraftModelProposer` takes ownership of the lifecycle and
        // pushes the path/dtype through the trait surface directly.
        self.load_draft_model_metal()
            .map_err(|e| ::vllm_engine::spec_decode::BackendError::Backend(e.to_string()))?;
        Ok(::vllm_engine::spec_decode::ModelHandle(1))
    }

    fn allocate_kv_pool(
        &mut self,
        model: ::vllm_engine::spec_decode::ModelHandle,
        num_blocks: usize,
    ) -> Result<::vllm_engine::spec_decode::KvPoolHandle, ::vllm_engine::spec_decode::BackendError>
    {
        if model != ::vllm_engine::spec_decode::ModelHandle(1) {
            return Err(::vllm_engine::spec_decode::BackendError::UnknownHandle(
                "ModelHandle (only the draft handle 1 supports allocate_kv_pool today)",
            ));
        }
        self.initialize_draft_cache_metal(num_blocks)
            .map_err(|e| ::vllm_engine::spec_decode::BackendError::Backend(e.to_string()))?;
        Ok(::vllm_engine::spec_decode::KvPoolHandle(1))
    }

    fn kv_per_block_bytes(
        &self,
        model: ::vllm_engine::spec_decode::ModelHandle,
    ) -> Result<usize, ::vllm_engine::spec_decode::BackendError> {
        let model_ref: &dyn ferrite_forward::FerriteWeights = match model {
            ::vllm_engine::spec_decode::ModelHandle::TARGET => {
                self.model.as_deref().ok_or_else(|| {
                    ::vllm_engine::spec_decode::BackendError::Backend(
                        "target model not loaded".into(),
                    )
                })?
            }
            ::vllm_engine::spec_decode::ModelHandle(1) => {
                self.draft_model.as_deref().ok_or_else(|| {
                    ::vllm_engine::spec_decode::BackendError::Backend(
                        "draft model not loaded".into(),
                    )
                })?
            }
            _ => {
                return Err(::vllm_engine::spec_decode::BackendError::UnknownHandle(
                    "ModelHandle",
                ));
            }
        };
        Ok(kv_per_block_bytes(model_ref, self.config.block_size))
    }
}

#[cfg(feature = "metal")]
impl Worker for FerriteWorker {
    fn init_device(&mut self) -> ExecutorResult<()> {
        let device = ferrite_metal_kernels::detect_device()
            .ok_or_else(|| ExecutorError::WorkerInit("no Metal device available".to_string()))?;
        self.metal_device = Some(std::sync::Arc::new(device));
        info!("FerriteWorker(metal): Metal device initialized");
        Ok(())
    }

    fn load_model(&mut self) -> ExecutorResult<()> {
        let _t_total = std::time::Instant::now();
        let t_resolve = std::time::Instant::now();
        let model_dir = resolve_model_path(
            &self.config.model_path,
            self.config.hf_token.as_deref(),
            self.config.gguf_file.as_deref(),
        )?;
        info!(
            "FerriteWorker(metal): resolved model dir in {:?} ({})",
            t_resolve.elapsed(),
            model_dir.display()
        );

        // Metal currently supports safetensors only — GGUF requires the
        // cuda-side dequant kernels. Surface that early with a clear
        // message rather than failing deep in `GpuWeights::from_dir`.
        if model_dir.is_file() && model_dir.extension().is_some_and(|e| e == "gguf") {
            return Err(ExecutorError::WorkerInit(
                "GGUF source not supported under metal (safetensors only)".into(),
            ));
        }

        let t_cfg = std::time::Instant::now();
        let hf_config = HfModelConfig::from_path(&model_dir)
            .map_err(|e| ExecutorError::WorkerInit(format!("config parse failed: {e}")))?;
        let arch = hf_config.architectures.first().cloned().unwrap_or_default();
        info!(
            "FerriteWorker(metal): parsed hf_config in {:?} (arch = {arch})",
            t_cfg.elapsed()
        );

        // Build the metal `GpuDevice` (device + queue + allocator). The
        // allocator goes inside `GpuWeights` (consumed by `from_dir`) and
        // a parallel `Arc<MetalAllocator>` rides on `GpuDevice` for
        // dummy-tensor allocation during profiling. Both wrap the same
        // underlying `metal::Device` (cheap ObjC retain).
        let metal_dev = self
            .metal_device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("metal device not initialized".into()))?;
        let device_arc = std::sync::Arc::new(metal_dev.device.clone());
        // Single allocator shared between `GpuWeights` (loader-side
        // bump arena) and `GpuDevice.allocator` (worker-side
        // `buffer_for` reverse-lookup at ICB-record time). `MetalAllocator::clone`
        // shares the underlying arena vec via `Arc<Mutex<…>>`, so a
        // weight allocated through the GpuWeights clone is reachable
        // via the GpuDevice clone — same `MTLBuffer`s, same offsets.
        let allocator = MetalAllocator::new((*device_arc).clone());
        let gpu_device = GpuDevice::new(device_arc.clone(), std::sync::Arc::new(allocator.clone()));
        let t_from_dir = std::time::Instant::now();
        let mut weights = GpuWeights::from_dir(&model_dir, allocator)
            .map_err(|e| ExecutorError::WorkerInit(format!("weight load failed: {e}")))?;
        info!(
            "FerriteWorker(metal): GpuWeights::from_dir in {:?} ({} tensors)",
            t_from_dir.elapsed(),
            weights.len()
        );

        // Metal backend default dtype is bf16 — matches the on-disk
        // `torch_dtype: bfloat16` of every modern HF Llama / Qwen /
        // Phi / Mistral checkpoint and the cuda backend's native
        // dtype. Apple Silicon M3+ has hardware bf16 MMA; the metal
        // shaders ship `_bf16_specialized` siblings for every
        // on-path kernel and `MetalDtype::Bf16` is the default on
        // `CanonicalParams::METAL_DTYPE`. F16-on-disk weights get
        // cast up to bf16 here (lossy in the mantissa but matches
        // the kernel's expected binding type).
        weights.set_target_dtype(GpuDType::BF16);

        // Build HF fingerprint — same disambiguation surface cuda uses.
        // `max_position_embeddings` is intentionally omitted (see
        // `HfFingerprint`'s doc-comment).
        let hf_fp = ferrite_forward::HfFingerprint {
            rope_scaling_type: hf_config
                .extra
                .get("rope_scaling")
                .and_then(|rs| rs.get("rope_type").or_else(|| rs.get("type")))
                .and_then(|v| v.as_str()),
            rope_scaling_hash: hf_config
                .extra
                .get("rope_scaling")
                .map(ferrite_forward::hash_json_value),
        };

        let max_model_len = self
            .config
            .max_model_len
            .or(hf_config.max_position_embeddings)
            .unwrap_or(4096);

        // Under cfg(metal), `CUstream = ()`; the per-canonical
        // `Weights::load` body ignores the stream parameter (uploads go
        // through the allocator inside `GpuWeights`).
        let t_try_load = std::time::Instant::now();
        let model = ferrite_forward::try_load(
            &mut weights,
            (),
            arch.as_str(),
            1, // tp_world_size — metal is tp=1 only
            0, // tp_rank
            max_model_len,
            hf_fp,
        )
        .map_err(|e| ExecutorError::WorkerInit(format!("ferrite-forward load: {e}")))?
        .ok_or_else(|| ExecutorError::ArchNotSupported(arch.clone()))?;

        info!(
            "FerriteWorker(metal): try_load in {:?} ({} via ferrite-forward, {})",
            t_try_load.elapsed(),
            arch,
            model.arch_name()
        );

        {
            use std::sync::atomic::Ordering;
            let s = weights.metal_allocator().load_stats();
            let zc = s.zero_copy_calls.load(Ordering::Relaxed);
            let zb = s.zero_copy_bytes.load(Ordering::Relaxed);
            let zcr = s.zero_copy_relaxed_calls.load(Ordering::Relaxed);
            let zbr = s.zero_copy_relaxed_bytes.load(Ordering::Relaxed);
            let mc = s.memcpy_calls.load(Ordering::Relaxed);
            let mb = s.memcpy_bytes.load(Ordering::Relaxed);
            let small = s.memcpy_small_calls.load(Ordering::Relaxed);
            let med = s.memcpy_med_calls.load(Ordering::Relaxed);
            let large = s.memcpy_large_calls.load(Ordering::Relaxed);
            let unaligned = s.memcpy_unaligned.load(Ordering::Relaxed);
            let outside = s.memcpy_outside_mmap.load(Ordering::Relaxed);
            info!(
                "FerriteWorker(metal): load routing — zero-copy {zc} calls / {:.1} MiB \
                 (of which {} calls / {:.1} MiB took the dtype-relaxed gate) | \
                 memcpy {mc} calls / {:.1} MiB ({} small <1MiB, {} med 1-16MiB, {} large ≥16MiB; \
                 fallback reason: {} unaligned, {} outside-mmap)",
                zb as f64 / (1 << 20) as f64,
                zcr,
                zbr as f64 / (1 << 20) as f64,
                mb as f64 / (1 << 20) as f64,
                small,
                med,
                large,
                unaligned,
                outside,
            );
            let hist: Vec<u64> = s
                .alignment_hist
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect();
            info!(
                "FerriteWorker(metal): tensor-offset alignment histogram \
                 (#trailing-zeros → count): {:?}",
                hist
            );
        }

        // Compile + cache the greedy-sampling pipeline once. Argmax fires
        // outside the per-bucket ICB, so it owns its own pipeline cache
        // here on the worker rather than living in the per-canonical
        // `MetalWorkerPool`.
        let t_argmax = std::time::Instant::now();
        let argmax = ferrite_metal_kernels::argmax::ArgmaxKernels::new(&gpu_device.device)
            .map_err(|e| ExecutorError::WorkerInit(format!("argmax kernel compile: {e:?}")))?;
        info!(
            "FerriteWorker(metal): ArgmaxKernels::new in {:?}",
            t_argmax.elapsed()
        );

        // Phase 6 chain-advance pipeline. Tiny kernel — one-time build
        // alongside argmax so the K-step chain driver never falls into
        // pipeline-compile latency on the first call.
        let t_chain = std::time::Instant::now();
        let chain_advance =
            ferrite_metal_kernels::chain_advance::ChainAdvanceKernel::new(&gpu_device.device)
                .map_err(|e| {
                    ExecutorError::WorkerInit(format!("chain_advance kernel compile: {e:?}"))
                })?;
        info!(
            "FerriteWorker(metal): ChainAdvanceKernel::new in {:?}",
            t_chain.elapsed()
        );

        self.gpu_device = Some(gpu_device);
        self.model = Some(model);
        self.argmax_kernels = Some(argmax);
        self.chain_advance_kernel = Some(chain_advance);
        self.model_dir = Some(model_dir);
        self.hf_config = Some(hf_config);
        self.resolved_architecture = Some(arch);

        // Phase 3 of DRAFT_SPEC_DECODE_PLAN.md: load the speculative
        // draft model onto the same gpu_device (shared allocator +
        // residency set + command queue). KV pool for the draft is
        // allocated separately in `initialize_cache`.
        if self.config.draft_model_path.is_some() {
            self.load_draft_model_metal()?;
        }

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
        let device = self
            .gpu_device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("gpu_device not initialized".into()))?;

        if self.config.kv_cache_dtype == "fp8_e4m3" || self.config.kv_cache_dtype == "fp8" {
            return Err(ExecutorError::WorkerInit(
                "FP8 KV cache not supported on metal — use F16".into(),
            ));
        }

        // Metal KV cache is f16; the per-canonical `forward` reads block
        // pointers via `KvCachePool::k_layer_mem` / `v_layer_mem` (added
        // in 3.E forward half) and binds them into the per-bucket
        // `RuntimeBindings.kv_cache_k/v` Buffer slots.
        let mtl_device = device.device.clone();
        // Mirror the model's resolved dtype so the KvCachePool's
        // tensor labels match what the kernel reads. bf16 weights →
        // bf16 cache; f16 weights → f16 cache. Byte size is the same
        // (2 bytes/elt) so this is purely a metadata fix; the
        // attention kernel binds the right `_bf16_specialized` /
        // `_f16_specialized` variant via `model.metal_dtype()`.
        let cache_dtype = match model.metal_dtype() {
            ferrite_forward::interpreter::metal::MetalDtype::Bf16 => GpuDType::BF16,
            ferrite_forward::interpreter::metal::MetalDtype::F16 => GpuDType::F16,
            ferrite_forward::interpreter::metal::MetalDtype::Int4 => {
                return Err(ExecutorError::WorkerInit(
                    "KV cache cannot be int4-quantized — must be bf16/f16".into(),
                ));
            }
        };

        // Hand newly-created KV cache buffers to the SHARED residency
        // set the allocator already owns (the same set covers the
        // weight arenas, every per-worker arena, and now KV). This is
        // the layout MLX uses — one `MTLResidencySet` per device queue,
        // not one per buffer category. Two residency sets per queue
        // (the previous layout) produced non-deterministic decode
        // output on Llama-3.2 even though every kernel passed its
        // golden in isolation.
        //
        // KvCachePool buffers are StorageModePrivate (~7.8 GiB total
        // for Llama-3.2-3B at 28 layers × 2 K/V × ~140 MiB), which
        // is exactly the working-set Apple's lazy paging tracker
        // drops out of residency under pressure. Without pinning,
        // attention reads race against the pager.
        let residency = device.allocator.residency().clone();

        let t_pool = std::time::Instant::now();
        let pool = unsafe {
            KvCachePool::new(
                model.num_hidden_layers() as usize,
                num_gpu_blocks,
                self.config.block_size,
                model.num_key_value_heads() as usize,
                model.head_dim() as usize,
                cache_dtype,
                |bytes| {
                    // KV cache is GPU-only (no CPU touches between
                    // forwards). StorageModePrivate avoids the
                    // unified-memory first-touch cost that
                    // StorageModeShared pays on each fresh cmdbuf
                    // (~3s/forward observed at TinyLlama).
                    let buffer = mtl_device
                        .newBufferWithLength_options(
                            bytes,
                            ::objc2_metal::MTLResourceOptions::StorageModePrivate,
                        )
                        .expect("newBufferWithLength_options returned nil");
                    residency.insert(&buffer);
                    Ok(RawGpuMem::from_buffer(buffer))
                },
            )
        }
        .map_err(|e| ExecutorError::WorkerInit(format!("KvCachePool: {e}")))?;
        info!(
            "FerriteWorker(metal): init_cache phases — KvCachePool::new(56 buffers) {:?} (residency.commit deferred to first forward)",
            t_pool.elapsed(),
        );
        // Attach the shared residency set to the device's queue so
        // every cmdbuf sees both arenas + KV-cache as wired. The
        // pool's lazy attach in `forward()` will commit the queued
        // KV inserts and attach the set on the first forward — moves
        // ~125ms of residency.commit() out of the init path. Attach
        // is idempotent per (queue, set) so the lazy path is safe
        // even if some other call has already attached.

        info!(
            "FerriteWorker(metal): KV cache initialized: {} layers × {} blocks × {} tokens",
            model.num_hidden_layers(),
            num_gpu_blocks,
            self.config.block_size,
        );
        self.kv_cache = Some(pool);

        // Phase 3 of DRAFT_SPEC_DECODE_PLAN.md: allocate the draft model's
        // KV pool alongside the target's, sized to fit inside the same
        // shared `recommendedMaxWorkingSetSize` budget.
        if self.draft_model.is_some() {
            self.initialize_draft_cache_metal(num_gpu_blocks)?;
        }

        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        // Apple unified memory: `recommendedMaxWorkingSetSize` is Apple's
        // own recommended budget for resident MTLBuffers (typically ~75%
        // of physical RAM on M-series, accounting for OS reservations).
        // `currentAllocatedSize` covers everything Metal has allocated
        // for this process so far — model weights live there
        // post-`load_model`.
        //
        // Peak activation: the metal pool's per-worker arena is sized
        // for the elementwise-max across every bucket spec
        // (`for_buckets` does the max). We expose that per-canonical
        // sum via `FerriteWeights::metal_arena_peak_bytes()` (emitted
        // by the per-canonical macro from the colored slot map at
        // expansion time). For metal we currently run at most
        // `max_workers = 1`, so peak == arena_peak. Runtime bindings
        // (input_ids/positions/etc.) and the per-step staging buffers
        // we allocate inside `execute_model` are <10 MiB and dwarfed
        // by the arena; we add a 64 MiB pad as a conservative bound.
        let metal_device = self
            .metal_device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("metal device not initialized".into()))?;
        let total = metal_device.device.recommendedMaxWorkingSetSize() as usize;
        let weights_and_overhead = metal_device.device.currentAllocatedSize();
        let arena_peak = self
            .model
            .as_ref()
            .map(|m| m.metal_arena_peak_bytes() as usize)
            .unwrap_or(512 * 1024 * 1024);
        // When a draft model is loaded we also need an activation arena for
        // it. Decode-only forwards on a 1B model peak well below the target,
        // but the target's prefill (M >> 1) arena is the worst case across
        // the pair — use it for both as a safe upper bound.
        let arena_peak_pair = if self.draft_model.is_some() {
            arena_peak.saturating_mul(2)
        } else {
            arena_peak
        };
        let peak_activation_estimate = arena_peak_pair.saturating_add(64 * 1024 * 1024);
        let utilization = self.config.gpu_memory_utilization;
        let available = compute_available_kv_bytes(
            total,
            weights_and_overhead,
            peak_activation_estimate,
            utilization,
        );
        // Per-pair split: when a draft model is loaded, every target KV
        // block has a 1:1 mirror in the draft pool, so the engine should
        // think it has only `target / (target + draft)` of the budget.
        // After the engine divides by `target_per_block_bytes` to land on
        // num_gpu_blocks, the same count of draft blocks fits inside the
        // remaining `draft / (target + draft)` slice.
        let (available_reported, draft_reservation) = if let Some(draft) = self.draft_model.as_ref()
        {
            let target = self.model.as_ref().expect("model loaded before draft");
            let bs = self.config.block_size;
            let t_pb = kv_per_block_bytes(target.as_ref(), bs);
            let d_pb = kv_per_block_bytes(draft.as_ref(), bs);
            let denom = t_pb.saturating_add(d_pb).max(1);
            // available * t_pb / (t_pb + d_pb), in u128 to dodge overflow.
            let scaled = (available as u128 * t_pb as u128 / denom as u128) as usize;
            (scaled, available.saturating_sub(scaled))
        } else {
            (available, 0)
        };
        info!(
            "FerriteWorker(metal): total={:.1} GiB, weights+overhead={:.1} GiB, \
             arena_peak={:.1} MiB (pair={:.1} MiB), kv_budget={:.1} GiB \
             (target_share={:.1} GiB, draft_reserve={:.1} GiB)",
            total as f64 / 1_073_741_824.0,
            weights_and_overhead as f64 / 1_073_741_824.0,
            arena_peak as f64 / 1_048_576.0,
            arena_peak_pair as f64 / 1_048_576.0,
            available as f64 / 1_073_741_824.0,
            available_reported as f64 / 1_073_741_824.0,
            draft_reservation as f64 / 1_073_741_824.0,
        );
        Ok(available_reported)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        // ── 1. Lifecycle: drop finished requests ──────────────────
        for req_id in &scheduler_output.finished_req_ids {
            self.token_buffers.remove(req_id);
            self.prompt_lengths.remove(req_id);
            self.annotation_buffers.remove(req_id);
            self.mm_data_buffers.remove(req_id);
            self.sampling_params_map.remove(req_id);
            self.seeded_rngs.remove(req_id);
        }
        self.input_batch
            .remove_finished(&scheduler_output.finished_req_ids);

        // ── 2. Lifecycle: add newly scheduled requests ────────────
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
            self.prompt_lengths
                .insert(new_req.req_id.clone(), prompt_ids.len());
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

        // ── 3. Lifecycle: refresh cached requests' block tables ───
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

        if self.input_batch.num_active() == 0 {
            return Ok(ModelRunnerOutput::empty());
        }

        // ── 4. Prepare flat batch inputs ─────────────────────────
        // Phase 4.6: forward the scheduler's spec drafts into
        // `prepare_inputs` so verify batches get the
        // `[last_token, draft_0..K-1]` flat shape per req. For non-spec
        // batches `scheduled_spec_decode_tokens` is empty → existing
        // q_len=1 path is unchanged.
        let mut prepared = self
            .input_batch
            .prepare_inputs(&scheduler_output.scheduled_spec_decode_tokens);
        let attn = &prepared.attn_meta;
        let num_tokens = attn.total_tokens;
        let num_reqs = attn.num_reqs;
        let block_size = self.config.block_size;

        // slot_mapping[t] = block_ids[abs_pos / bs] * bs + (abs_pos % bs)
        // u32 under metal — the macro-emitted forward reads this as
        // `&[u32]`. Rope kernel checks `slot_mapping[i] != u32::MAX`.
        let mut slot_mapping_u32: Vec<u32> = Vec::with_capacity(num_tokens);
        for i in 0..num_reqs {
            let tokens_before = attn.tokens_before[i];
            let q_len = attn.q_lens[i];
            let block_ids = &attn.block_ids[i];
            for t in 0..q_len {
                let abs_pos = tokens_before + t;
                let block_idx = abs_pos / block_size;
                let offset = abs_pos % block_size;
                if block_idx < block_ids.len() {
                    slot_mapping_u32.push((block_ids[block_idx] * block_size + offset) as u32);
                } else {
                    slot_mapping_u32.push(u32::MAX);
                }
            }
        }

        // block_table padded to [num_reqs, max_blocks_per_seq] u32.
        //
        // Row stride MUST match the ferrite-metal kernel's
        // `ATTN_PAGED_MAX_BLOCKS_PER_SEQ` function constant (slot 5,
        // baked from `W::MAX_BLOCKS_PER_SEQ` — default 128 in
        // `instr.rs::CanonicalParams`). The kernel reads
        // `block_table + seq_idx * MAX_BLOCKS_PER_SEQ`; if the host
        // writes with a smaller `runtime_max_blocks` stride, every
        // `seq_idx > 0` reads from the wrong row offset and pulls
        // garbage K/V. Symptom: concurrent / batched-decode
        // requests other than seq_idx=0 produce incoherent output;
        // single-seq runs are unaffected because only row 0 is read.
        //
        // Plumbing the per-canonical `MAX_BLOCKS_PER_SEQ` through
        // the worker would require the FerriteWeights trait to
        // expose it; for now hardcode the trait default (128). The
        // kernel constant 5 is set to this same value in
        // `lowering.rs::AttentionPrefillPaged` /
        // `AttentionViaCache` arms.
        const KERNEL_BLOCK_TABLE_STRIDE: usize = 128;
        let runtime_max_blocks = attn.block_ids.iter().map(|b| b.len()).max().unwrap_or(0);
        let max_blocks_eff = KERNEL_BLOCK_TABLE_STRIDE.max(runtime_max_blocks).max(1);
        let mut block_table_u32: Vec<u32> = vec![0u32; num_reqs * max_blocks_eff];
        if runtime_max_blocks > 0 {
            for (i, blocks) in attn.block_ids.iter().enumerate() {
                for (j, &bid) in blocks.iter().enumerate() {
                    block_table_u32[i * max_blocks_eff + j] = bid as u32;
                }
            }
        }

        let cu_seqlens_u32: Vec<u32> = attn.query_start_loc.iter().map(|&v| v as u32).collect();
        let seqused_k_u32: Vec<u32> = attn.seq_lens.iter().map(|&v| v as u32).collect();

        let max_seqlen_q = attn.q_lens.iter().copied().max().unwrap_or(0);
        let max_seqlen_k = attn.seq_lens.iter().copied().max().unwrap_or(0);
        let sample_indices: Vec<u32> = attn.sample_indices();
        let req_ids_in_order: Vec<String> = attn.req_ids.clone();
        let q_lens: Vec<usize> = attn.q_lens.clone();

        let input_ids_u32 = std::mem::take(&mut prepared.flat_token_ids);
        let positions_u32 = std::mem::take(&mut prepared.flat_positions);

        // ── 6. Forward + per-row argmax via SpecDecodeBackend trait ─
        //
        // Phase 5.2a routes the verify forward through
        // `forward_argmax_blocking`. This loses the fused-CB argmax
        // optimization (~0.5 ms / 5% per step) — phase 6 re-introduces
        // it via a dedicated trait primitive. The K-step draft chain
        // below stays inline until phase 5.4 moves it host-side into
        // `DraftModelProposer::propose`.
        // Spec verify? Any req carrying spec_token_ids flips the gate
        // so the lm_head slice trio (OnlyIfSingleSeqNoSpec) skips and
        // the full GEMM (OnlyIfMultiSeqOrSpec) writes every row of
        // logits — required for rejection sampling on > 1 sample
        // positions per seq.
        let has_spec_tokens = prepared
            .req_inputs
            .iter()
            .any(|r| !r.spec_token_ids.is_empty());

        // Phase 8: when a draft model is loaded AND we're in lockstep+K
        // mode (extended-batch disabled), kick off the lockstep prefill
        // on the dedicated `draft_queue` in a scoped thread, in
        // parallel with target verify on the main queue. Target writes
        // target-KV, lockstep writes draft-KV — separate KV pools, no
        // contention. Saves ~15-20 ms / step on the draft chain
        // critical path.
        let phase8_parallel_lockstep = self.draft_model.is_some()
            && self.draft_kv_cache.is_some()
            && self.draft_queue.is_some()
            && std::env::var_os("FERRITE_DRAFT_EXTENDED_BATCH").is_none();
        let mut async_lockstep_done = false;
        // Phase-9 results from the lockstep thread (populated when
        // enabled + eligible; empty otherwise).
        let mut spec9_seeds_final: Vec<u32> = Vec::new();
        let mut spec9_drafts_final: Vec<Vec<u32>> = Vec::new();
        let argmax_vec: Vec<u32> = {
            use ::vllm_engine::spec_decode::{
                ForwardArgmaxRequest, KvPoolHandle, ModelHandle, SpecDecodeBackend,
            };
            let req = ForwardArgmaxRequest {
                input_ids: &input_ids_u32,
                positions: &positions_u32,
                slot_mapping: &slot_mapping_u32,
                cu_seqlens_q: &cu_seqlens_u32,
                seqused_k: &seqused_k_u32,
                block_table: &block_table_u32,
                block_table_stride: max_blocks_eff,
                max_seqlen_q,
                max_seqlen_k,
                num_tokens,
                has_spec_tokens,
            };
            // Phase 9 gating: enable the worker-side speculative
            // K-step chain (runs in the lockstep thread, overlapped
            // with target verify). Gated by env var until baseline
            // bench validates the projected ~13% TPOT win.
            let phase9_speculative_chain =
                phase8_parallel_lockstep && std::env::var_os("FERRITE_SPEC9_ENABLE").is_some();

            // Spec-9 eligibility, single-req case only for now.
            // Derive K from q_lens[0] (= drafts + 1 bonus when the
            // req was spec-decoded last step; = 1 otherwise). Skip
            // first-step / multi-req / chunked-prefill cases — they
            // fall through to the regular chain in the proposer.
            let spec9_single_req_eligible: bool = phase9_speculative_chain
                && num_reqs == 1
                && prepared
                    .req_inputs
                    .first()
                    .is_some_and(|r| !r.spec_token_ids.is_empty())
                && q_lens.first().copied().unwrap_or(0) >= 2;
            let spec9_k: usize = if spec9_single_req_eligible {
                q_lens[0] - 1
            } else {
                0
            };
            let spec9_tokens_before: usize = if spec9_single_req_eligible {
                attn.tokens_before[0]
            } else {
                0
            };
            let spec9_block_ids: Vec<u32> = if spec9_single_req_eligible {
                attn.block_ids[0].iter().map(|&b| b as u32).collect()
            } else {
                Vec::new()
            };
            let spec9_block_size: usize = self.config.block_size;

            if phase8_parallel_lockstep {
                // Snapshot of inputs the lockstep thread needs.
                // Raw-pointer borrow of draft_model / draft_kv_cache is
                // sound here: both are accessed ONLY by the lockstep
                // thread (target verify uses self.model + self.kv_cache,
                // disjoint fields) and the thread can't outlive the
                // scope (scoped threads are joined before scope exit).
                // Decompose the fat pointer to `dyn FerriteWeights`
                // into two `usize` so the closure stays `Send`.
                // `*const dyn T` is `[data, vtable]` on the supported
                // targets; transmute to extract both.
                let dm_pair: [usize; 2] = {
                    let fat: *const dyn ferrite_forward::FerriteWeights =
                        self.draft_model.as_deref().expect("checked above");
                    unsafe {
                        std::mem::transmute::<*const dyn ferrite_forward::FerriteWeights, [usize; 2]>(
                            fat,
                        )
                    }
                };
                let dkv_addr: usize = (self.draft_kv_cache.as_ref().expect("checked above")
                    as *const KvCachePool) as usize;
                // Spec-9: raw pointers for argmax + chain_advance
                // kernels so the speculative chain can call
                // `metal_chain_dispatch` from inside the thread
                // (no `&self` access while target verify is in flight).
                let spec9_argmax_addr: usize = self.argmax_kernels.as_ref().map_or(0, |k| {
                    k as *const ferrite_metal_kernels::argmax::ArgmaxKernels as usize
                });
                let spec9_chain_addr: usize = self.chain_advance_kernel.as_ref().map_or(0, |k| {
                    k as *const ferrite_metal_kernels::chain_advance::ChainAdvanceKernel as usize
                });
                let main_dev = self.gpu_device.as_ref().expect("init");
                let mtl_device_clone = main_dev.device.clone();
                let allocator_clone = main_dev.allocator.clone();
                let draft_queue_clone = self.draft_queue.as_ref().expect("checked above").clone();
                // Snapshot host slices the lockstep thread reads.
                let lock_input_ids = &input_ids_u32;
                let lock_positions = &positions_u32;
                let lock_slot_mapping = &slot_mapping_u32;
                let lock_cu_seqlens = &cu_seqlens_u32;
                let lock_seqused_k = &seqused_k_u32;
                let lock_block_table = &block_table_u32;
                let lock_block_table_stride = max_blocks_eff;
                let lock_max_q = max_seqlen_q;
                let lock_max_k = max_seqlen_k;
                let lock_num_tokens = num_tokens;
                let lock_num_reqs = num_reqs;

                // Stage-1 spec9 hypothesis probe: capture draft's
                // last-row-per-req argmax inside the scoped lockstep
                // thread so we can compare to target's actual bonus
                // after the join. Pure measurement — no behavioral
                // change yet.
                let lock_cu_for_thread = lock_cu_seqlens.to_vec();
                // Spec-9 captures (moved into thread closure).
                let spec9_eligible_thread = spec9_single_req_eligible;
                let spec9_k_thread = spec9_k;
                let spec9_tokens_before_thread = spec9_tokens_before;
                let spec9_block_ids_thread = spec9_block_ids.clone();
                let spec9_block_size_thread = spec9_block_size;
                let spec9_block_table_stride_thread = lock_block_table_stride;
                let (v, spec9_seeds_out, spec9_drafts_out): (Vec<u32>, Vec<u32>, Vec<Vec<u32>>) =
                    std::thread::scope(|s| {
                        let lockstep_handle =
                        s.spawn(move || -> (Vec<u32>, Vec<Vec<u32>>) {
                        // Build shadow GpuDevice on draft_queue.
                        let mut shadow_device = GpuDevice {
                            device: mtl_device_clone,
                            queue: draft_queue_clone,
                            allocator: allocator_clone,
                        };
                        // Upload host slices into fresh shared
                        // MTLBuffers (thread-local, dropped at thread
                        // exit).
                        let dev = &shadow_device.device;
                        let b_in = Self::alloc_shared_u32_buf(dev, lock_input_ids);
                        let b_pos = Self::alloc_shared_u32_buf(dev, lock_positions);
                        let b_slot = Self::alloc_shared_u32_buf(dev, lock_slot_mapping);
                        let b_cu = Self::alloc_shared_u32_buf(dev, lock_cu_seqlens);
                        let b_su = Self::alloc_shared_u32_buf(dev, lock_seqused_k);
                        let b_bt = Self::alloc_shared_u32_buf(dev, lock_block_table);

                        let dt = ferrite_cuda_core::dtype::DType::U32;
                        let v_in = unsafe { TensorView::from_raw(GpuTensor::new(
                            b_in.contents().as_ptr() as *mut u8,
                            &[lock_num_tokens.max(1)], dt)) };
                        let v_pos = unsafe { TensorView::from_raw(GpuTensor::new(
                            b_pos.contents().as_ptr() as *mut u8,
                            &[lock_num_tokens.max(1)], dt)) };
                        let v_slot = unsafe { TensorView::from_raw(GpuTensor::new(
                            b_slot.contents().as_ptr() as *mut u8,
                            &[lock_num_tokens.max(1)], dt)) };
                        let v_cu = unsafe { TensorView::from_raw(GpuTensor::new(
                            b_cu.contents().as_ptr() as *mut u8,
                            &[lock_cu_seqlens.len().max(1)], dt)) };
                        let v_su = unsafe { TensorView::from_raw(GpuTensor::new(
                            b_su.contents().as_ptr() as *mut u8,
                            &[lock_seqused_k.len().max(1)], dt)) };
                        let v_bt = unsafe { TensorView::from_raw(GpuTensor::new(
                            b_bt.contents().as_ptr() as *mut u8,
                            &[lock_num_reqs.max(1), lock_block_table_stride.max(1)], dt)) };

                        // Reassemble fat pointer from (data, vtable)
                        // pair, then immutable-borrow.
                        let dm_fat_reassembled: *const dyn ferrite_forward::FerriteWeights =
                            unsafe {
                                std::mem::transmute::<
                                    [usize; 2],
                                    *const dyn ferrite_forward::FerriteWeights,
                                >(dm_pair)
                            };
                        let dm = unsafe { &*dm_fat_reassembled };
                        let dkv: &KvCachePool =
                            unsafe { &*(dkv_addr as *const KvCachePool) };
                        let ctx = ferrite_forward::ForwardCtx {
                            input_ids: v_in,
                            positions: v_pos,
                            slot_mapping: v_slot,
                            cu_seqlens_q: v_cu,
                            seqused_k: v_su,
                            block_table: v_bt,
                            max_seqlen_q: lock_max_q,
                            max_seqlen_k: lock_max_k,
                            kv_cache: dkv,
                            has_spec_tokens: false,
                        };
                        let logits = unsafe {
                            dm.forward(&ctx, &mut shadow_device, lock_num_tokens as u64)
                        };

                        // Phase-9: argmax of the last verify position
                        // per req IS draft's prediction of what target
                        // will sample as the step's bonus token.
                        // Computed when EITHER the probe OR the
                        // speculative chain is enabled. Bf16 only
                        // (matches `metal_dtype()` for our test
                        // models).
                        let need_spec_seed =
                            std::env::var_os("FERRITE_SPEC9_PROBE").is_some()
                                || spec9_eligible_thread;
                        let spec_seeds: Vec<u32> = if need_spec_seed {
                            let logits_ptr =
                                logits.as_gpu_tensor().raw_ptr() as *const u8;
                            let vocab = dm.vocab_size() as usize;
                            let mut out = Vec::with_capacity(lock_num_reqs);
                            for i in 0..lock_num_reqs {
                                let last_row =
                                    lock_cu_for_thread[i + 1] as usize - 1;
                                let row_off = last_row * vocab * 2;
                                let row = unsafe {
                                    std::slice::from_raw_parts(
                                        logits_ptr.add(row_off) as *const u16,
                                        vocab,
                                    )
                                };
                                let mut best_idx: u32 = 0;
                                let mut best_val: f32 = f32::NEG_INFINITY;
                                for (j, &bits) in row.iter().enumerate() {
                                    let v = f32::from_bits((bits as u32) << 16);
                                    if v > best_val {
                                        best_val = v;
                                        best_idx = j as u32;
                                    }
                                }
                                out.push(best_idx);
                            }
                            out
                        } else {
                            Vec::new()
                        };

                        // Phase-9 speculative chain (single-req case):
                        // dispatch the K-step chain on draft_queue using
                        // spec_seeds[0] as the iter-0 input_ids. Runs
                        // serially after lockstep on draft_queue, but
                        // in parallel with target verify on main_queue.
                        // If target's sampled bonus matches spec_seed
                        // AND target accepted all K drafts, the proposer
                        // returns these drafts directly and the chain
                        // is hidden behind target verify. Otherwise,
                        // proposer falls back to running the chain
                        // with the corrected seed (re-overwrites
                        // draft KV at the now-correct positions —
                        // see `project-spec-decode-phase6-handoff`).
                        let spec_drafts: Vec<Vec<u32>> = if spec9_eligible_thread
                            && !spec_seeds.is_empty()
                        {
                            let argmax_kernels_ref: &ferrite_metal_kernels::argmax::ArgmaxKernels = unsafe {
                                &*(spec9_argmax_addr
                                    as *const ferrite_metal_kernels::argmax::ArgmaxKernels)
                            };
                            let chain_kernel_ref: &ferrite_metal_kernels::chain_advance::ChainAdvanceKernel = unsafe {
                                &*(spec9_chain_addr
                                    as *const ferrite_metal_kernels::chain_advance::ChainAdvanceKernel)
                            };
                            // Single-req inputs assuming "all K drafts
                            // accepted + bonus": position after bonus =
                            // tokens_before + (q_lens-1) + 1 + 1 ...
                            // Actually simpler — the K-step chain's
                            // iter-0 position is `tokens_before + q_lens`
                            // (= the slot after the bonus that target
                            // will sample). For was_spec_decode this
                            // ASSUMES all K drafts accepted (the
                            // "all-hit" speculative case the proposer
                            // validates).
                            let chain_pos = spec9_tokens_before_thread
                                + spec9_k_thread + 1;
                            let block_idx = chain_pos / spec9_block_size_thread;
                            let offset = chain_pos % spec9_block_size_thread;
                            let slot = if block_idx
                                < spec9_block_ids_thread.len()
                            {
                                spec9_block_ids_thread[block_idx]
                                    * spec9_block_size_thread as u32
                                    + offset as u32
                            } else {
                                u32::MAX
                            };
                            let spec_input_ids: Vec<u32> = vec![spec_seeds[0]];
                            let spec_positions: Vec<u32> = vec![chain_pos as u32];
                            let spec_slot: Vec<u32> = vec![slot];
                            let spec_cu: Vec<u32> = vec![0, 1];
                            let spec_su: Vec<u32> = vec![(chain_pos + 1) as u32];
                            let spec_req =
                                ::vllm_engine::spec_decode::ForwardArgmaxRequest {
                                    input_ids: &spec_input_ids,
                                    positions: &spec_positions,
                                    slot_mapping: &spec_slot,
                                    cu_seqlens_q: &spec_cu,
                                    seqused_k: &spec_su,
                                    block_table: lock_block_table,
                                    block_table_stride:
                                        spec9_block_table_stride_thread,
                                    max_seqlen_q: 1,
                                    max_seqlen_k: chain_pos + 1,
                                    num_tokens: 1,
                                    has_spec_tokens: false,
                                };
                            metal_chain_dispatch(
                                dm,
                                dkv,
                                &mut shadow_device,
                                argmax_kernels_ref,
                                chain_kernel_ref,
                                &spec_req,
                                spec9_block_size_thread,
                                spec9_k_thread,
                            )
                            .unwrap_or_else(|e| {
                                if std::env::var_os("FERRITE_SPEC9_PROBE")
                                    .is_some()
                                {
                                    eprintln!(
                                        "[spec9] speculative chain dispatch \
                                         failed (falling back): {e}"
                                    );
                                }
                                Vec::new()
                            })
                        } else {
                            Vec::new()
                        };

                        (spec_seeds, spec_drafts)
                    });

                        let r = self
                            .forward_argmax_blocking(
                                ModelHandle::TARGET,
                                KvPoolHandle::TARGET,
                                &req,
                            )
                            .map_err(|e| {
                                ExecutorError::WorkerExecution(format!("spec verify forward: {e}"))
                            });
                        // join propagates any panic the lockstep thread
                        // raised; treat as worker execution failure.
                        let (spec_seeds, spec_drafts) =
                            lockstep_handle.join().expect("lockstep thread panicked");
                        let target_argmax = r.unwrap_or_else(|_| Vec::new());
                        if !spec_seeds.is_empty() && !target_argmax.is_empty() {
                            // Probe: compare draft's predicted bonus
                            // vs target's actual bonus (per req).
                            // cu_seqlens_q[i+1]-1 is the last verify row.
                            let cu = &lock_cu_seqlens;
                            for i in 0..lock_num_reqs {
                                let target_bonus_idx = cu[i + 1] as usize - 1;
                                if target_bonus_idx < target_argmax.len() {
                                    let target_bonus = target_argmax[target_bonus_idx];
                                    let spec_seed = spec_seeds[i];
                                    eprintln!(
                                        "[spec9-probe] req={} spec_seed={} \
                                     target_bonus={} match={}",
                                        i,
                                        spec_seed,
                                        target_bonus,
                                        spec_seed == target_bonus,
                                    );
                                }
                            }
                        }
                        (target_argmax, spec_seeds, spec_drafts)
                    });
                async_lockstep_done = true;
                spec9_seeds_final = spec9_seeds_out;
                spec9_drafts_final = spec9_drafts_out;
                v
            } else {
                self.forward_argmax_blocking(ModelHandle::TARGET, KvPoolHandle::TARGET, &req)
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("spec verify forward: {e}"))
                    })?
            }
        };
        let total_n = argmax_vec.len() as u32;
        let argmax_slice: &[u32] = &argmax_vec;

        // ── 6.6. Per-request rejection sampling (phase 4.6) ──
        //
        // Build `sampled_token_ids` + `was_spec_decode` BEFORE the
        // draft phase so the K-step chain can seed from accepted
        // (= argmax under greedy) tokens instead of raw target argmax.
        // For non-spec batches (`ReqSlice.spec_token_ids` empty), grab
        // the single argmax at the req's last sample position. For
        // verify batches (K drafts), run `greedy_rejection_sample`
        // over the K+1 contiguous argmax rows.
        let mut sampled_token_ids: Vec<Vec<u32>> = Vec::with_capacity(num_reqs);
        let mut was_spec_decode: Vec<bool> = Vec::with_capacity(num_reqs);
        let mut req_id_to_index: std::collections::HashMap<String, usize> =
            std::collections::HashMap::with_capacity(num_reqs);
        for (i, req_id) in req_ids_in_order.iter().enumerate() {
            let req_slice = &prepared.req_inputs[i];
            req_id_to_index.insert(req_id.clone(), i);
            if req_slice.spec_token_ids.is_empty() {
                let row = sample_indices[i] as usize;
                debug_assert!(row < total_n as usize);
                sampled_token_ids.push(vec![argmax_slice[row]]);
                was_spec_decode.push(false);
            } else {
                let start = req_slice.token_start;
                let end = start + req_slice.token_count;
                debug_assert!(end <= total_n as usize);
                let target_ids = &argmax_slice[start..end];
                let rejection = ::vllm_engine::spec_decode::greedy_rejection_sample(
                    target_ids,
                    &req_slice.spec_token_ids,
                );
                sampled_token_ids.push(rejection.accepted_tokens);
                was_spec_decode.push(true);
            }
        }

        // ── 6.7. Draft seed bundle ──────────────────────────────
        //
        // Pre-5.4 the worker ran the lockstep prefill + K autoregressive
        // draft decode chain inline here against `self.draft_model` +
        // `self.draft_kv_cache`. As of phase 5.4 the chain lives in
        // `vllm_engine::spec_decode::DraftModelProposer::propose_for_step`
        // (host-side) and reaches the same GPU paths via the
        // `SpecDecodeBackend` trait. The worker just pre-packages the
        // owned-data the proposer needs into `draft_seed_inputs`; the
        // engine pulls `executor.spec_decode_backend()` and drives the
        // chain in `finalize_step`.
        let draft_seed_inputs: Option<::vllm_engine::spec_decode::DraftSeedInputs> =
            if self.draft_model.is_some() && self.draft_kv_cache.is_some() {
                Some(::vllm_engine::spec_decode::DraftSeedInputs {
                    input_ids: input_ids_u32,
                    positions: positions_u32,
                    slot_mapping: slot_mapping_u32,
                    cu_seqlens_q: cu_seqlens_u32,
                    seqused_k: seqused_k_u32,
                    block_table: block_table_u32,
                    block_table_stride: max_blocks_eff,
                    max_seqlen_q,
                    max_seqlen_k,
                    num_tokens,
                    req_ids: req_ids_in_order.clone(),
                    block_ids: attn
                        .block_ids
                        .iter()
                        .map(|v| v.iter().map(|&b| b as u32).collect())
                        .collect(),
                    tokens_before: attn.tokens_before.clone(),
                    q_lens: q_lens.clone(),
                    was_spec_decode: was_spec_decode.clone(),
                    block_size,
                    // Phase 8: when target verify ran in parallel
                    // with lockstep prefill on draft_queue (above),
                    // the worker has already waited for both — the
                    // proposer skips its in-proposer lockstep call.
                    async_lockstep_done,
                    speculative_seeds: spec9_seeds_final.clone(),
                    speculative_chain_drafts: spec9_drafts_final.clone(),
                })
            } else {
                None
            };

        // ── 7. Commit per-request state ─────────────────────────
        for (i, req_id) in req_ids_in_order.iter().enumerate() {
            let q_len = q_lens[i];
            self.input_batch
                .commit_step(req_id, &sampled_token_ids[i], q_len, was_spec_decode[i]);
        }

        Ok(ModelRunnerOutput {
            req_ids: req_ids_in_order,
            req_id_to_index,
            sampled_token_ids,
            logprobs: None,
            prompt_logprobs_dict: std::collections::HashMap::new(),
            // K-step draft chain runs host-side now; the proposer fills
            // `set_spec_token_ids` directly. `draft_token_ids` stays
            // `None` from the worker.
            draft_token_ids: None,
            draft_seed_inputs,
            pooler_output: None,
            d2h_resolver: None,
        })
    }

    fn compile_or_warm_up_model(&mut self) -> ExecutorResult<()> {
        // Metal pipelines are JIT-compiled lazily via the MetalWorkerPool's
        // function-constant cache; no eager warmup needed for Step 2.
        Ok(())
    }

    fn shutdown(&mut self) {
        self.is_shutdown = true;
        // Drop order matters: KV cache references device buffers; model
        // holds Weights backed by `GpuWeights` whose allocator arenas
        // back every weight tensor. Drop tensors before device.
        self.kv_cache = None;
        self.model = None;
        self.argmax_kernels = None;
        self.gpu_device = None;
        self.metal_device = None;
    }

    fn rank(&self) -> usize {
        self.config.tp_rank
    }

    fn local_rank(&self) -> usize {
        self.config.tp_rank
    }

    fn is_driver_worker(&self) -> bool {
        self.config.tp_rank == 0
    }

    fn take_preloaded_tokenizer(&mut self) -> Option<tokenizers::Tokenizer> {
        self.preloaded_tokenizer.take()
    }

    fn architecture(&self) -> Option<String> {
        self.resolved_architecture.clone()
    }

    fn spec_decode_backend(
        &mut self,
    ) -> Option<&mut dyn vllm_engine::spec_decode::SpecDecodeBackend> {
        Some(self as &mut dyn vllm_engine::spec_decode::SpecDecodeBackend)
    }
}
