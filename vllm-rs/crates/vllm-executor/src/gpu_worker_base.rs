// SPDX-License-Identifier: Apache-2.0
//! Shared GPU worker utilities.
//!
//! Common logic used by all GPU-based `Worker` implementations (FerriteWorker,
//! TkWorkerAdapter, etc.): memory estimation. Hand-written CUDA model forwards
//! were removed; ferrite-forward owns model construction now, including its
//! own HF-config parsing.
//!
//! Mirrors Python vLLM's `Worker` class which handles shared GPU plumbing
//! while delegating model-specific execution to `GPUModelRunner`.

// ---------------------------------------------------------------------------
// Model path resolution lives on `ferrite_worker::resolve_model_path` —
// this module previously held a copy that diverged. Both copies have
// been consolidated onto the ferrite_worker version (the one with
// parallel shard download + GGUF auto-detect + HF-Cache short-circuit).
// `vllm-serve` and the worker `load_model` bodies all call into the
// canonical version directly.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Memory estimation helpers
// ---------------------------------------------------------------------------

/// Compute available bytes for KV cache given GPU memory profile data.
///
/// Matches Python vLLM's computation:
///   available = total * utilization - weights_and_overhead - peak_activations - redundancy
pub fn compute_available_kv_bytes(
    total_memory: usize,
    weights_and_overhead: usize,
    peak_activations: usize,
    gpu_memory_utilization: f64,
) -> usize {
    let redundancy_buffer: usize = 150 * 1024 * 1024; // 150 MiB
    let non_kv_cache = weights_and_overhead + peak_activations + redundancy_buffer;
    let requested = (total_memory as f64 * gpu_memory_utilization) as usize;
    requested.saturating_sub(non_kv_cache)
}
