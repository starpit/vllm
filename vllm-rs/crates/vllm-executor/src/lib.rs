// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Worker/process management and distributed coordination.
//!
//! This crate provides:
//!
//! * [`worker`] -- The `Worker` trait defining the device-level execution interface.
//!   Port of `vllm/v1/worker/worker_base.py`.
//! * [`uniproc`] -- `UniProcExecutor`: single-process executor wrapping one worker.
//!   Port of `vllm/v1/executor/uniproc_executor.py`.
//! * [`multiproc`] -- `MultiprocExecutor`: multi-process executor managing worker processes.
//!   Port of `vllm/v1/executor/multiproc_executor.py`.
//! * [`parallel`] -- Distributed parallel state types (TP/PP groups, rank management).
//! * [`error`] -- Executor-specific error types.

pub mod candle_worker;
pub mod error;
pub mod multiproc;
pub mod parallel;
pub mod uniproc;
pub mod worker;
