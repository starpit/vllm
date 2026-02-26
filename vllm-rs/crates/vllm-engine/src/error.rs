// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Engine-specific error types.

/// Errors that can occur in the engine core.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// Scheduler error.
    #[error("scheduler error: {0}")]
    Scheduler(String),

    /// Executor error.
    #[error("executor error: {0}")]
    Executor(String),

    /// Request not found.
    #[error("request not found: {0}")]
    RequestNotFound(String),

    /// Engine is shut down.
    #[error("engine is shut down")]
    Shutdown,

    /// Protocol/transport error.
    #[error("transport error: {0}")]
    Transport(#[from] vllm_protocol::transport::TransportError),

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),
}

pub type EngineResult<T> = Result<T, EngineError>;
