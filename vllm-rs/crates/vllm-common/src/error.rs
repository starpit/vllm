// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Error types for the vLLM Rust port.

use thiserror::Error;

/// Top-level error type for vllm-common operations.
#[derive(Debug, Error)]
pub enum VllmError {
    /// A validation error for a specific parameter.
    #[error("Validation error for parameter '{parameter}': {message}")]
    Validation {
        message: String,
        parameter: String,
        value: String,
    },

    /// A request was not found.
    #[error("Request not found: {request_id}")]
    RequestNotFound { request_id: String },

    /// An engine-level error.
    #[error("Engine error: {0}")]
    Engine(String),

    /// A scheduling error.
    #[error("Scheduler error: {0}")]
    Scheduler(String),

    /// A serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// An internal error (catch-all).
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Convenience alias used throughout the crate.
pub type VllmResult<T> = Result<T, VllmError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error_display() {
        let err = VllmError::Validation {
            message: "temperature must be non-negative".into(),
            parameter: "temperature".into(),
            value: "-1.0".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("temperature"));
        assert!(msg.contains("non-negative"));
    }

    #[test]
    fn test_request_not_found_display() {
        let err = VllmError::RequestNotFound {
            request_id: "req-42".into(),
        };
        assert_eq!(format!("{err}"), "Request not found: req-42");
    }

    #[test]
    fn test_engine_error_display() {
        let err = VllmError::Engine("worker crashed".into());
        assert_eq!(format!("{err}"), "Engine error: worker crashed");
    }

    #[test]
    fn test_scheduler_error_display() {
        let err = VllmError::Scheduler("out of blocks".into());
        assert_eq!(format!("{err}"), "Scheduler error: out of blocks");
    }

    #[test]
    fn test_internal_error_display() {
        let err = VllmError::Internal("unexpected state".into());
        assert_eq!(format!("{err}"), "Internal error: unexpected state");
    }

    #[test]
    fn test_result_alias() {
        let ok: VllmResult<u32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);

        let err: VllmResult<u32> = Err(VllmError::Internal("boom".into()));
        assert!(err.is_err());
    }
}
