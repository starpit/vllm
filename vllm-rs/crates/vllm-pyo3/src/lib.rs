// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! PyO3 bridge module for vLLM Rust port.
//!
//! This crate exposes the Rust scheduler and KV cache manager to Python
//! via PyO3. It acts as the integration layer between the existing Python
//! vLLM engine and the new Rust components.
//!
//! Over time, as more components are ported to Rust, this crate will
//! shrink -- eventually becoming optional (only needed for models that
//! haven't been ported yet).

use pyo3::prelude::*;

use vllm_common::{Request, RequestStatus, SamplingParams};
use vllm_core::scheduler::{PauseState, Scheduler, SchedulerInterface};

// ---------------------------------------------------------------------------
// RustScheduler -- Python-visible wrapper around the Rust Scheduler
// ---------------------------------------------------------------------------

/// A Rust-native scheduler exposed to Python via PyO3.
///
/// This wraps the `vllm_core::scheduler::Scheduler` and provides methods
/// matching the Python `SchedulerInterface` so it can be used as a drop-in
/// replacement for the Python scheduler.
///
/// Usage from Python:
/// ```python
/// from vllm_rs import RustScheduler
/// sched = RustScheduler(
///     max_num_seqs=256,
///     max_num_batched_tokens=8192,
///     max_model_len=4096,
///     num_gpu_blocks=1024,
///     block_size=16,
///     policy="fcfs",
///     enable_chunked_prefill=True,
/// )
/// ```
#[pyclass(unsendable)]
struct RustScheduler {
    inner: Scheduler,
}

#[pymethods]
impl RustScheduler {
    /// Create a new RustScheduler.
    #[new]
    #[pyo3(signature = (
        max_num_seqs = 256,
        max_num_batched_tokens = 8192,
        max_model_len = 4096,
        num_gpu_blocks = 1024,
        block_size = 16,
        policy = "fcfs",
        enable_chunked_prefill = true,
        long_prefill_token_threshold = 0,
        max_num_scheduled_tokens = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_num_seqs: usize,
        max_num_batched_tokens: usize,
        max_model_len: usize,
        num_gpu_blocks: usize,
        block_size: usize,
        policy: &str,
        enable_chunked_prefill: bool,
        long_prefill_token_threshold: usize,
        max_num_scheduled_tokens: Option<usize>,
    ) -> PyResult<Self> {
        let sched_policy = match policy {
            "fcfs" => vllm_config::SchedulerPolicy::Fcfs,
            "priority" => vllm_config::SchedulerPolicy::Priority,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown scheduling policy: {other}"
                )));
            }
        };

        let config = vllm_config::SchedulerConfig {
            max_num_batched_tokens,
            max_num_seqs,
            max_num_scheduled_tokens,
            policy: sched_policy,
            enable_chunked_prefill,
            long_prefill_token_threshold,
            ..Default::default()
        };

        let sched =
            Scheduler::with_simple_blocks(&config, max_model_len, num_gpu_blocks, block_size);

        Ok(Self { inner: sched })
    }

    /// Add a new request to the scheduler.
    #[pyo3(signature = (request_id, prompt_token_ids, max_tokens = 16, arrival_time = 0.0, priority = 0))]
    fn add_request(
        &mut self,
        request_id: String,
        prompt_token_ids: Vec<u32>,
        max_tokens: u32,
        arrival_time: f64,
        priority: i32,
    ) {
        let mut params = SamplingParams::default();
        params.max_tokens = Some(max_tokens);

        let request = Request::new(
            request_id,
            prompt_token_ids,
            params,
            arrival_time,
            0, // client_index
            priority,
            None, // cache_salt
        );

        self.inner.add_request(request);
    }

    /// Run one scheduling step. Returns a dict with scheduling results.
    fn schedule(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let output = self.inner.schedule();

        let dict = pyo3::types::PyDict::new(py);

        // num_scheduled_tokens
        let tokens_dict = pyo3::types::PyDict::new(py);
        for (req_id, num_tokens) in &output.num_scheduled_tokens {
            tokens_dict.set_item(req_id, num_tokens)?;
        }
        dict.set_item("num_scheduled_tokens", tokens_dict)?;
        dict.set_item(
            "total_num_scheduled_tokens",
            output.total_num_scheduled_tokens,
        )?;

        // New request IDs
        let new_req_ids: Vec<&str> = output
            .scheduled_new_reqs
            .iter()
            .map(|r| r.req_id.as_str())
            .collect();
        dict.set_item("new_request_ids", new_req_ids)?;

        // Cached request IDs
        dict.set_item("cached_request_ids", &output.scheduled_cached_reqs.req_ids)?;

        // Finished request IDs
        let finished: Vec<&str> = output.finished_req_ids.iter().map(|s| s.as_str()).collect();
        dict.set_item("finished_req_ids", finished)?;

        Ok(dict.into())
    }

    /// Abort/finish requests by ID.
    #[pyo3(signature = (request_ids, status = "abort"))]
    fn finish_requests(&mut self, request_ids: Vec<String>, status: &str) -> Vec<(String, u32)> {
        let rs_status = match status {
            "abort" => RequestStatus::FinishedAborted,
            "stopped" => RequestStatus::FinishedStopped,
            "length" => RequestStatus::FinishedLengthCapped,
            "error" => RequestStatus::FinishedError,
            _ => RequestStatus::FinishedAborted,
        };

        let id_refs: Vec<&str> = request_ids.iter().map(|s| s.as_str()).collect();
        self.inner.finish_requests(&id_refs, rs_status)
    }

    /// Get the number of unfinished requests.
    fn get_num_unfinished_requests(&self) -> usize {
        self.inner.get_num_unfinished_requests()
    }

    /// Whether there are unfinished requests.
    fn has_unfinished_requests(&self) -> bool {
        self.inner.has_unfinished_requests()
    }

    /// Whether there are finished requests pending notification.
    fn has_finished_requests(&self) -> bool {
        self.inner.has_finished_requests()
    }

    /// Whether there are any requests (unfinished or pending-finished).
    fn has_requests(&self) -> bool {
        self.inner.has_requests()
    }

    /// Get request counts as (num_running, num_waiting).
    fn get_request_counts(&self) -> (usize, usize) {
        self.inner.get_request_counts()
    }

    /// Set the pause state (0=unpaused, 1=paused_new, 2=paused_all).
    fn set_pause_state(&mut self, state: u8) -> PyResult<()> {
        let ps = match state {
            0 => PauseState::Unpaused,
            1 => PauseState::PausedNew,
            2 => PauseState::PausedAll,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid pause state: {other}"
                )));
            }
        };
        self.inner.set_pause_state(ps);
        Ok(())
    }

    /// Get the current pause state (0, 1, or 2).
    fn pause_state(&self) -> u8 {
        self.inner.pause_state() as u8
    }

    /// Reset the prefix cache.
    fn reset_prefix_cache(&mut self) -> bool {
        self.inner.reset_prefix_cache()
    }

    /// Shut down the scheduler.
    fn shutdown(&mut self) {
        self.inner.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Python module definition
// ---------------------------------------------------------------------------

/// The `vllm_rs` Python module -- Rust-accelerated components for vLLM.
#[pymodule]
fn vllm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustScheduler>()?;
    Ok(())
}
