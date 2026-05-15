// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Executor-specific error types.

/// Errors that can occur in executor operations.
#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    /// Worker initialization failed.
    #[error("worker initialization failed: {0}")]
    WorkerInit(String),

    /// The worker backend has no implementation for this model
    /// architecture. Distinct from `WorkerInit` so a caller can
    /// fall back to a different backend (e.g. metal `FerriteWorker`
    /// → `MlxWorker` for arches without a ferrite-forward metal
    /// variant) without parsing the failure string.
    #[error("architecture `{0}` not supported by this backend")]
    ArchNotSupported(String),

    /// Worker execution failed.
    #[error("worker execution failed: {0}")]
    WorkerExecution(String),

    /// Worker is not healthy.
    #[error("worker health check failed: {0}")]
    WorkerUnhealthy(String),

    /// Worker process died unexpectedly.
    #[error("worker process died: rank {rank}")]
    WorkerDied { rank: usize },

    /// Communication error between executor and worker.
    #[error("communication error: {0}")]
    Communication(String),

    /// Executor is shut down.
    #[error("executor is shut down")]
    Shutdown,

    /// Invalid configuration.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// Timeout waiting for worker response.
    #[error("timeout waiting for worker (rank {rank}): {message}")]
    Timeout { rank: usize, message: String },

    /// Engine error (forwarded from vllm-engine).
    #[error("engine error: {0}")]
    Engine(#[from] vllm_engine::error::EngineError),
}

pub type ExecutorResult<T> = Result<T, ExecutorError>;
