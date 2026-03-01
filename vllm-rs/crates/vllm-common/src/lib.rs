// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm-common` -- shared types, errors, and utilities for the vLLM Rust port.
//!
//! This crate provides the foundational types that are used across all other
//! vLLM Rust crates:
//!
//! * [`error`] -- Error types (`VllmError`, `VllmResult`).
//! * [`sampling`] -- Sampling parameters and related enums.
//! * [`request`] -- The `Request` struct and `RequestStatus` enum.
//! * [`engine_io`] -- Engine-core I/O types (`EngineCoreRequest`,
//!   `EngineCoreOutput`, `EngineCoreOutputs`, events, `FinishReason`).

pub mod engine_io;
pub mod error;
pub mod multimodal;
pub mod request;
pub mod sampling;
pub mod telemetry;

// ---- Convenience re-exports ------------------------------------------------
// These allow downstream crates to write `use vllm_common::SamplingParams;`
// instead of `use vllm_common::sampling::SamplingParams;`.

pub use engine_io::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest,
    FinishReason, SchedulerStats, StopReason,
};
pub use error::{VllmError, VllmResult};
pub use multimodal::MultimodalData;
pub use request::{Request, RequestStatus};
pub use sampling::{LogprobsOutput, RequestOutputKind, SamplingParams, SamplingType, TokenLogprob};
