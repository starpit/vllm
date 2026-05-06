// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `MetalWorker`: a `Worker` implementation using ferrite-metal for forwards.
//!
//! # ⚠️ INTERIM — STEPPING STONE TOWARD CUDA/METAL UNIFICATION ⚠️
//!
//! **Read this before extending the file.**
//!
//! This file deliberately duplicates a subset of [`super::cuda_worker::CudaWorker`]
//! rather than sharing code with it. The duplication is **temporary**.
//!
//! ## Why a separate file at all
//!
//! `CudaWorker` is ~9k lines and runs only under `feature = "cuda"`. Most of it
//! is generic (request lifecycle, scheduler-output translation, sampling-param
//! resolution, grammar, pooling, HF download) — but ~30% is CUDA-coupled
//! (h2d/d2h, argmax kernel, init_device, build_attention_tensors) and another
//! ~15% is CUDA-only (CUDA graphs, NCCL TP, pipeline parallel) or
//! legacy-arch fallback that ferrite-models is gradually replacing.
//!
//! Two paths to a backend-generic Worker were considered:
//!
//! - **Top-down extraction:** lift the generic logic out of `cuda_worker.rs`
//!   into a `worker_common` module, generify over a `Backend` trait, instantiate
//!   for both backends. Cleaner endpoint but requires landing a 3.5k-LOC
//!   refactor on a no-CUDA host (this worktree) — high risk of cuda regressions
//!   we can't catch locally.
//!
//! - **Bottom-up duplication (chosen):** build a focused `MetalWorker` (this
//!   file), get the metal end-to-end golden green, *then* extract the shared
//!   shape using two working impls as the anchor. Interim duplication is real,
//!   but cheap to delete once we know the right cut points from experience
//!   instead of audit.
//!
//! ## What this file should NOT grow into
//!
//! Keep this file **smaller** than `cuda_worker.rs`. Specifically, **do not** add:
//!
//! - CUDA graph capture / piecewise execution (Metal has ICBs already).
//! - NCCL / pipeline parallel — ferrite-metal is single-device for now.
//! - Per-arch hand-written model dispatch — only the ferrite path. Arches that
//!   ferrite-metal doesn't compile yet fall back to `MlxWorker` in
//!   `vllm-serve/src/init.rs`, not here.
//! - Spec decoding, multimodal, FP8 KV — defer until the basic golden lands.
//!
//! Anything that would also be added to `CudaWorker` belongs in a shared module
//! once unification (the U.1 task) starts.
//!
//! ## Unification target
//!
//! When U.1 lands, the shared shape is expected to look like:
//!
//! ```ignore
//! struct WorkerCore<B: Backend> {
//!     // generic fields: input_batch, token_buffers, sampling_params, …
//! }
//!
//! impl<B: Backend> WorkerCore<B> {
//!     fn execute_model_inner(&mut self, ...) -> ExecutorResult<...> {
//!         // generic body, calls B::h2d, B::forward, B::sample, B::d2h
//!     }
//! }
//!
//! pub struct CudaWorker { core: WorkerCore<CudaBackend>, /* cuda-only extras */ }
//! pub struct MetalWorker { core: WorkerCore<MetalBackend> }
//! ```
//!
//! Until then, treat changes here as candidates for the eventual extraction:
//! when you copy logic from `CudaWorker`, copy it as-close-to-verbatim as
//! possible — divergent code is harder to merge later than parallel code.
//!
//! See `FERRITE_METAL_PROGRESS.md` (M.* phase) and
//! `project_ferrite_metal_port_handoff.md` for context.

#![cfg(feature = "metal")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tracing::info;
use vllm_common::SamplingParams;
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_engine::executor::ModelRunnerOutput;
use vllm_model::weight::HfModelConfig;

use crate::error::{ExecutorError, ExecutorResult};
use crate::input_batch::InputBatch;
use crate::worker::Worker;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a `MetalWorker`.
///
/// Mirrors the subset of [`super::cuda_worker::CudaWorkerConfig`] that's
/// meaningful on Apple Silicon. Fields with no metal analog
/// (cuda_graph_mode, tp_*, pp_*, gguf_file, kv_cache_dtype=fp8) are absent
/// rather than ignored — adding them silently would mask "this isn't wired"
/// at the API boundary.
#[derive(Debug, Clone)]
pub struct MetalWorkerConfig {
    /// Path to a local model directory, or a HuggingFace model ID.
    pub model_path: String,
    /// Data type for model weights: "auto", "f16", "bf16".
    pub dtype: String,
    /// Optional HuggingFace token for gated models.
    pub hf_token: Option<String>,
    /// KV cache block size in tokens (must match the scheduler's block size).
    pub block_size: usize,
    /// Maximum tokens per scheduler iteration (controls runtime buffer sizing).
    pub max_num_batched_tokens: usize,
    /// Fraction of unified memory to use (0.0-1.0). Used for KV cache budget.
    /// On unified-memory Apple Silicon "GPU memory" is not a hard partition;
    /// this controls how aggressively we claim system RAM for KV blocks.
    pub gpu_memory_utilization: f64,
    /// Pooling strategy: "auto", "last", "cls", "mean".
    pub pooling_strategy: String,
    /// Whether the worker runs in pooling mode (--runner pooling).
    pub is_pooling: bool,
    /// Maximum sequence length the engine should honour.
    pub max_model_len: Option<usize>,
    /// EOS token IDs — threaded into seal-pad detection at sampling time.
    pub eos_token_ids: Vec<u32>,
}

// ---------------------------------------------------------------------------
// MetalWorker
// ---------------------------------------------------------------------------

/// `Worker` implementation backed by ferrite-metal's `MetalWorkerPool`.
///
/// State is intentionally a trimmed subset of [`super::cuda_worker::CudaWorker`].
/// See module-level doc for the unification plan.
///
/// `dead_code` allowed: M.1 lays down the field shape; M.2/M.3/M.4 read each
/// field as the load/cache/execute paths fill in. Removing fields here just
/// to silence the warning would force re-adding them next chunk — better to
/// keep the shape stable and let the compiler flag if a field ends up unused
/// past the M.* phase.
#[allow(dead_code)]
pub struct MetalWorker {
    config: MetalWorkerConfig,
    /// Loaded HuggingFace config (populated by `load_model`).
    hf_config: Option<HfModelConfig>,
    /// Resolved model directory on local disk.
    model_dir: Option<PathBuf>,
    /// Resolved architecture string from `config.json`'s `architectures[0]`.
    resolved_architecture: Option<String>,
    /// Tokenizer parsed during model load (consumed by the engine).
    preloaded_tokenizer: Option<tokenizers::Tokenizer>,

    // --- Loaded model + pool (M.2 will populate) ---
    /// The arch-erased model handle; one variant per ferrite-supported arch.
    model: Option<MetalModel>,

    // --- Generic batch state (shared with CudaWorker — already in vllm-executor) ---
    input_batch: InputBatch,
    token_buffers: HashMap<String, Vec<u32>>,
    prompt_lengths: HashMap<String, usize>,
    sampling_params_map: HashMap<String, SamplingParams>,
    seeded_rngs: HashMap<String, rand::rngs::StdRng>,

    // --- KV cache config (M.3 will populate) ---
    num_gpu_blocks: usize,

    // --- Lifecycle ---
    is_shutdown: bool,
}

/// Arch-erased loaded model. Each ferrite-metal-supported arch is one variant.
///
/// This enum stays small on purpose — the macro emits per-canonical
/// `Weights` + `metal_pool` + `load` symbols; the unification with CUDA's
/// `FerriteModel` (a `Box<dyn FerriteWeights>`) will collapse this to a
/// single trait-object once the metal side learns the same dyn-trait shape.
/// For now, the per-canonical route lets us avoid re-doing dyn-dispatch
/// machinery before the basic golden is green.
///
/// **Empty for M.1.** M.2 fills in `TinyLlama` first, then other arches.
#[allow(dead_code)] // populated by M.2.
enum MetalModel {
    /// Placeholder — keeps the enum non-empty so match-exhaustiveness works.
    /// Replaced by real arch arms in M.2 (`TinyLlama1_1b(...)`, etc.).
    Unimplemented,
}

impl MetalWorker {
    /// Construct a fresh `MetalWorker`. No GPU work happens here — call
    /// [`Worker::init_device`] then [`Worker::load_model`] to bring it online.
    pub fn new(config: MetalWorkerConfig) -> Self {
        Self {
            config,
            hf_config: None,
            model_dir: None,
            resolved_architecture: None,
            preloaded_tokenizer: None,
            model: None,
            input_batch: InputBatch::new(),
            token_buffers: HashMap::new(),
            prompt_lengths: HashMap::new(),
            sampling_params_map: HashMap::new(),
            seeded_rngs: HashMap::new(),
            num_gpu_blocks: 0,
            is_shutdown: false,
        }
    }

    /// Loaded HF config (None until `load_model` runs).
    pub fn hf_config(&self) -> Option<&HfModelConfig> {
        self.hf_config.as_ref()
    }

    /// Local model directory (None until `load_model` runs).
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Worker trait impl
// ---------------------------------------------------------------------------

impl Worker for MetalWorker {
    fn init_device(&mut self) -> ExecutorResult<()> {
        // Apple Silicon's unified memory + Metal device discovery happens
        // lazily inside `MetalWorkerPool::for_buckets` (M.2). No per-thread
        // context to set, no graph runner to allocate. Symmetric to
        // `MlxWorker::init_device` in vllm-mlx — both are no-ops.
        info!("MetalWorker: initialized (Apple Silicon GPU, ferrite-metal backend)");
        Ok(())
    }

    fn load_model(&mut self) -> ExecutorResult<()> {
        // Filled in by M.2: HF download → parse config → resolve arch →
        // build GpuWeights via MetalAllocator → call per-arch `load(...)`
        // → build `MetalWorkerPool` → store in `self.model`.
        Err(ExecutorError::WorkerInit(
            "MetalWorker::load_model not yet implemented (M.2)".into(),
        ))
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        _num_cpu_blocks: usize,
    ) -> ExecutorResult<()> {
        // Filled in by M.3: allocate the paged KV block-pool in unified
        // memory and hand the block-table machinery to the model handle.
        // For now, just record the figure so the engine keeps moving
        // through the lifecycle.
        self.num_gpu_blocks = num_gpu_blocks;
        info!(
            "MetalWorker: cache budget recorded ({} blocks); pool wiring pending (M.3)",
            num_gpu_blocks
        );
        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        // Apple Silicon uses unified memory — GPU and CPU share the same pool.
        // Mirror `MlxWorker::determine_available_memory`: report total physical
        // RAM via `sysctl HW_MEMSIZE`; the OS pages out what we don't touch.
        // The `gpu_memory_utilization` factor is applied by the executor in
        // `compute_available_kv_bytes`, not here.
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
            // sysctl failed (vanishingly rare on macOS). Fall back to a
            // conservative 8 GiB rather than 0 (which would zero the KV
            // budget and crash the engine).
            return Ok(8 * 1024 * 1024 * 1024);
        }
        Ok(total)
    }

    fn execute_model(
        &mut self,
        _scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        // Filled in by M.4: translate SchedulerOutput → ForwardInputs,
        // call `MetalWorkerPool::forward`, run greedy argmax, d2h
        // sampled tokens, build ModelRunnerOutput.
        Err(ExecutorError::WorkerExecution(
            "MetalWorker::execute_model not yet implemented (M.4)".into(),
        ))
    }

    fn take_preloaded_tokenizer(&mut self) -> Option<tokenizers::Tokenizer> {
        self.preloaded_tokenizer.take()
    }

    fn architecture(&self) -> Option<String> {
        self.resolved_architecture.clone()
    }

    fn shutdown(&mut self) {
        self.is_shutdown = true;
        // M.2 will replace this with a clean drop of the pool/model handle.
        // Today there's nothing live to release.
    }

    fn rank(&self) -> usize {
        // Single-device on Apple Silicon — no TP/PP, always rank 0. If
        // multi-device support arrives, plumb through `MetalWorkerConfig`.
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

    fn make_config() -> MetalWorkerConfig {
        MetalWorkerConfig {
            model_path: "test-model".to_string(),
            dtype: "auto".to_string(),
            hf_token: None,
            block_size: 16,
            max_num_batched_tokens: 1024,
            gpu_memory_utilization: 0.9,
            pooling_strategy: "auto".to_string(),
            is_pooling: false,
            max_model_len: Some(2048),
            eos_token_ids: vec![],
        }
    }

    #[test]
    fn new_worker_is_uninitialized() {
        let w = MetalWorker::new(make_config());
        assert!(w.hf_config().is_none());
        assert!(w.model_dir().is_none());
        assert_eq!(w.rank(), 0);
        assert_eq!(w.local_rank(), 0);
        assert!(w.is_driver_worker());
    }

    #[test]
    fn init_device_is_noop() {
        let mut w = MetalWorker::new(make_config());
        w.init_device().expect("init_device is a no-op");
    }

    #[test]
    fn determine_available_memory_returns_nonzero() {
        let mut w = MetalWorker::new(make_config());
        let mem = w
            .determine_available_memory()
            .expect("sysctl HW_MEMSIZE should succeed on macOS");
        assert!(mem >= 1024 * 1024 * 1024, "mem = {mem}, expected >= 1 GiB");
    }

    #[test]
    fn load_model_returns_pending_error() {
        // M.2 replaces this — the test catches accidental "Ok(())" stubs that
        // would silently skip model loading.
        let mut w = MetalWorker::new(make_config());
        let err = w.load_model().expect_err("M.2 not implemented yet");
        let msg = format!("{err:?}");
        assert!(msg.contains("M.2"), "error should reference task M.2: {msg}");
    }

    // No `execute_model` test in M.1 — `SchedulerOutput` has non-trivial
    // construction (no `Default` impl) and the cuda side covers the
    // request-lifecycle wiring. M.4 will add an integration test that
    // builds a real `SchedulerOutput` against TinyLlama-1.1B.
}
