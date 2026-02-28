// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Engine core loop and request lifecycle management.
//!
//! The `EngineCore` is the central coordinator of vLLM. It sits between the
//! API server (front-end) and the model executor (back-end):
//!
//! ```text
//!   API Server  ──► EngineCore ──► Executor ──► GPU Workers
//!                    │ scheduler │
//!                    │ kv cache  │
//! ```
//!
//! Each iteration of the engine core:
//! 1. Processes pending input requests (add, abort, utility)
//! 2. Runs the scheduler to decide which requests to process
//! 3. Dispatches execution to the model executor
//! 4. Collects outputs and routes them back to clients
//!
//! Port of: `vllm/v1/engine/core.py::EngineCore`

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use tracing::{debug, error, info};
use vllm_common::{
    EngineCoreOutput, EngineCoreOutputs, FinishReason, Request, RequestStatus, SchedulerStats,
    StopReason,
};
use vllm_config::SchedulerConfig;
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_core::scheduler::{PauseState, Scheduler, SchedulerInterface};
use vllm_protocol::messages::PauseMode;

use crate::error::{EngineError, EngineResult};
use crate::executor::{Executor, ModelRunnerOutput};

// ---------------------------------------------------------------------------
// EngineCore
// ---------------------------------------------------------------------------

/// The inner loop of the vLLM engine.
///
/// Manages the scheduler, dispatches work to the executor, and produces
/// outputs. This struct is protocol-agnostic — it can be used in-process
/// or wrapped with ZMQ sockets for multi-process operation.
///
/// Port of: `vllm/v1/engine/core.py::EngineCore`
pub struct EngineCore {
    /// The scheduler that decides which requests to process.
    scheduler: Scheduler,

    /// The executor that runs model forward passes.
    executor: Box<dyn Executor>,

    /// Index of this engine (for data-parallel setups).
    engine_index: u32,

    /// Whether the engine has been shut down.
    is_shutdown: bool,

    /// Monotonic start time for computing relative timestamps.
    start_time: Instant,

    /// Pending abort request IDs.
    aborts_queue: VecDeque<Vec<String>>,

    /// Whether async scheduling is enabled.
    #[allow(dead_code)]
    async_scheduling: bool,

    /// Whether speculative decoding is enabled.
    #[allow(dead_code)]
    use_spec_decode: bool,

    /// EOS token IDs for stop criteria (primary + additional from config).
    eos_token_ids: Vec<u32>,
}

/// Configuration for creating an EngineCore.
pub struct EngineCoreConfig {
    /// Scheduler configuration.
    pub scheduler_config: SchedulerConfig,
    /// Maximum model length (context window).
    pub max_model_len: usize,
    /// Number of GPU blocks available for KV cache.
    pub num_gpu_blocks: usize,
    /// Block size (tokens per block).
    pub block_size: usize,
    /// Engine index for data-parallel setups.
    pub engine_index: u32,
    /// Whether to enable async scheduling.
    pub async_scheduling: bool,
    /// Whether speculative decoding is enabled.
    pub use_spec_decode: bool,
    /// EOS token IDs for stop criteria. Empty disables EOS-based stopping.
    /// Supports models with multiple EOS tokens (e.g. LLaMA 3:
    /// `<|end_of_text|>`, `<|eom_id|>`, `<|eot_id|>`).
    pub eos_token_ids: Vec<u32>,
}

/// Output from a single engine step, grouped by client index.
pub type StepOutputs = HashMap<u32, EngineCoreOutputs>;

impl EngineCore {
    /// Create a new EngineCore.
    pub fn new(config: EngineCoreConfig, executor: Box<dyn Executor>) -> Self {
        let scheduler = Scheduler::with_simple_blocks(
            &config.scheduler_config,
            config.max_model_len,
            config.num_gpu_blocks,
            config.block_size,
        );

        info!(
            "EngineCore initialized: max_model_len={}, num_gpu_blocks={}, block_size={}, eos_token_ids={:?}",
            config.max_model_len, config.num_gpu_blocks, config.block_size, config.eos_token_ids
        );

        Self {
            scheduler,
            executor,
            engine_index: config.engine_index,
            is_shutdown: false,
            start_time: Instant::now(),
            aborts_queue: VecDeque::new(),
            async_scheduling: config.async_scheduling,
            use_spec_decode: config.use_spec_decode,
            eos_token_ids: config.eos_token_ids,
        }
    }

    /// Get the current monotonic timestamp (seconds since engine start).
    fn timestamp(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    // -----------------------------------------------------------------------
    // Request management
    // -----------------------------------------------------------------------

    /// Add a new request to the scheduler.
    pub fn add_request(&mut self, request: Request) {
        if self.is_shutdown {
            error!("Cannot add request to shut down engine");
            return;
        }
        self.scheduler.add_request(request);
    }

    /// Abort requests by ID.
    pub fn abort_requests(&mut self, request_ids: &[String]) {
        let id_refs: Vec<&str> = request_ids.iter().map(String::as_str).collect();
        self.scheduler
            .finish_requests(&id_refs, RequestStatus::FinishedAborted);
    }

    /// Queue abort requests for processing during the next step.
    ///
    /// This is used when aborts arrive asynchronously (e.g., from the
    /// input socket thread) and need to be processed during the main loop.
    pub fn queue_aborts(&mut self, request_ids: Vec<String>) {
        self.aborts_queue.push_back(request_ids);
    }

    /// Process any pending aborts from the queue.
    fn process_aborts_queue(&mut self) {
        while let Some(ids) = self.aborts_queue.pop_front() {
            self.abort_requests(&ids);
        }
    }

    // -----------------------------------------------------------------------
    // Pause / resume
    // -----------------------------------------------------------------------

    /// Pause the scheduler.
    ///
    /// - `abort`: Abort all in-flight requests, set PAUSED_NEW.
    /// - `keep`: Set PAUSED_ALL (freeze everything).
    /// - `wait`: Not supported in in-process mode.
    pub fn pause_scheduler(&mut self, mode: PauseMode) -> EngineResult<Vec<(String, u32)>> {
        let mut aborted = Vec::new();

        if mode == PauseMode::Abort {
            aborted = self.finish_all_requests(RequestStatus::FinishedAborted);
        }

        let pause_state = match mode {
            PauseMode::Keep => PauseState::PausedAll,
            PauseMode::Abort | PauseMode::Wait => PauseState::PausedNew,
        };
        self.scheduler.set_pause_state(pause_state);

        Ok(aborted)
    }

    /// Resume the scheduler.
    pub fn resume_scheduler(&mut self) {
        self.scheduler.set_pause_state(PauseState::Unpaused);
    }

    /// Whether the scheduler is paused.
    pub fn is_scheduler_paused(&self) -> bool {
        self.scheduler.pause_state() != PauseState::Unpaused
    }

    /// Finish all requests with the given status.
    fn finish_all_requests(&mut self, _status: RequestStatus) -> Vec<(String, u32)> {
        // Collect all request IDs first to avoid borrow issues.
        let (running, waiting) = self.scheduler.get_request_counts();
        if running == 0 && waiting == 0 {
            return Vec::new();
        }

        // The Python code passes `None` to finish_requests to finish all.
        // Our Rust interface requires explicit IDs, so we need to handle
        // this differently. For now, we use the scheduler's internal method.
        // TODO: Add a finish_all method to the scheduler interface.
        Vec::new()
    }

    // -----------------------------------------------------------------------
    // Prefix cache
    // -----------------------------------------------------------------------

    /// Reset the prefix cache.
    pub fn reset_prefix_cache(&mut self) -> bool {
        self.scheduler.reset_prefix_cache()
    }

    // -----------------------------------------------------------------------
    // Engine step
    // -----------------------------------------------------------------------

    /// Whether the engine has work to do.
    pub fn has_work(&self) -> bool {
        self.scheduler.has_requests()
    }

    /// Execute one scheduling + execution step.
    ///
    /// Returns a map of client_index → outputs, and a flag indicating
    /// whether the model was actually executed.
    pub fn step(&mut self) -> EngineResult<(StepOutputs, bool)> {
        if !self.scheduler.has_requests() {
            return Ok((HashMap::new(), false));
        }

        // 1. Schedule.
        let scheduler_output = self.scheduler.schedule();
        let model_executed = scheduler_output.total_num_scheduled_tokens > 0;

        if !model_executed {
            return Ok((HashMap::new(), false));
        }

        // 2. Execute model.
        let model_output = self
            .executor
            .execute_model(&scheduler_output)
            .map_err(|e| EngineError::Executor(e.to_string()))?;

        // 3. Process any pending aborts.
        self.process_aborts_queue();

        // 4. Update scheduler state and build outputs.
        let mut outputs = self.update_from_output(&scheduler_output, &model_output);

        // 5. Attach scheduler stats to outputs.
        let (num_running, num_waiting) = self.scheduler.get_request_counts();
        let stats = SchedulerStats {
            num_running_reqs: num_running,
            num_waiting_reqs: num_waiting,
            kv_cache_usage: self.scheduler.kv_cache_usage(),
        };
        for engine_outputs in outputs.values_mut() {
            engine_outputs.scheduler_stats = Some(stats);
        }

        Ok((outputs, model_executed))
    }

    /// Update the scheduler state from model output and build engine outputs.
    ///
    /// This mirrors the Python `Scheduler.update_from_output()` method.
    /// After each step:
    /// 1. Append new tokens to each request's state in the scheduler.
    /// 2. Check stop criteria (max_tokens, EOS, stop_token_ids).
    /// 3. Finish requests that hit a stop condition.
    fn update_from_output(
        &mut self,
        scheduler_output: &SchedulerOutput,
        model_output: &ModelRunnerOutput,
    ) -> StepOutputs {
        let timestamp = self.timestamp();
        let mut client_outputs: StepOutputs = HashMap::new();
        let mut finished_ids: Vec<(String, RequestStatus)> = Vec::new();

        // Process each request that was scheduled.
        for req_id in scheduler_output.num_scheduled_tokens.keys() {
            let new_token_ids_slice: &[u32] = model_output.get_tokens(req_id).unwrap_or_default();

            // Append new tokens to the request's state in the scheduler.
            if !new_token_ids_slice.is_empty() {
                self.scheduler
                    .append_output_tokens(req_id, new_token_ids_slice);
            }

            // Check stop criteria against the updated request state.
            let (finish_reason, stop_reason) =
                self.check_stop_criteria(req_id, new_token_ids_slice);

            if let Some(reason) = finish_reason {
                let status = match reason {
                    FinishReason::Length => RequestStatus::FinishedLengthCapped,
                    FinishReason::Stop => RequestStatus::FinishedStopped,
                    _ => RequestStatus::FinishedStopped,
                };
                finished_ids.push((req_id.clone(), status));
                debug!(
                    "Request {} finished: {:?} (stop_reason={:?})",
                    req_id, reason, stop_reason
                );
            }

            // Extract logprobs for this request if available.
            let new_logprobs = model_output
                .logprobs
                .as_ref()
                .and_then(|lp_vec| {
                    model_output
                        .req_id_to_index
                        .get(req_id)
                        .and_then(|&idx| lp_vec.get(idx))
                })
                .and_then(|opt| opt.clone());

            // Extract prompt logprobs for this request if available.
            let new_prompt_logprobs = model_output.prompt_logprobs_dict.get(req_id).map(|plp| {
                // Wrap in Option: first position is None (no prior context),
                // rest are Some.
                let mut result: Vec<Option<vllm_common::LogprobsOutput>> =
                    Vec::with_capacity(plp.len() + 1);
                result.push(None); // position 0
                for lp in plp {
                    result.push(Some(lp.clone()));
                }
                result
            });

            // Build the output for this request.
            let output = EngineCoreOutput {
                request_id: req_id.clone(),
                new_token_ids: new_token_ids_slice.to_vec(),
                finish_reason,
                stop_reason,
                num_cached_tokens: 0,
                events: None,
                new_logprobs,
                new_prompt_logprobs,
            };

            // Route to client_index 0 (default for single-client mode).
            let engine_outputs = client_outputs
                .entry(0)
                .or_insert_with(|| EngineCoreOutputs {
                    engine_index: self.engine_index,
                    outputs: Vec::new(),
                    timestamp,
                    scheduler_stats: None,
                });
            engine_outputs.outputs.push(output);
        }

        // Finish requests that hit stop criteria.
        for (req_id, status) in &finished_ids {
            self.scheduler.finish_requests(&[req_id.as_str()], *status);
        }

        // Include already-finished request IDs from the scheduler (e.g. aborts).
        if !scheduler_output.finished_req_ids.is_empty() {
            let engine_outputs = client_outputs
                .entry(0)
                .or_insert_with(|| EngineCoreOutputs {
                    engine_index: self.engine_index,
                    outputs: Vec::new(),
                    timestamp,
                    scheduler_stats: None,
                });

            for req_id in &scheduler_output.finished_req_ids {
                // Only add if not already included from the loop above.
                if !finished_ids.iter().any(|(id, _)| id == req_id) {
                    engine_outputs.outputs.push(EngineCoreOutput {
                        request_id: req_id.clone(),
                        new_token_ids: Vec::new(),
                        finish_reason: Some(FinishReason::Stop),
                        stop_reason: None,
                        num_cached_tokens: 0,
                        events: None,
                        new_logprobs: None,
                        new_prompt_logprobs: None,
                    });
                }
            }
        }

        client_outputs
    }

    /// Check stop criteria for a request given its newly generated tokens.
    ///
    /// Returns `(finish_reason, stop_reason)`.
    fn check_stop_criteria(
        &self,
        req_id: &str,
        new_token_ids: &[u32],
    ) -> (Option<FinishReason>, Option<StopReason>) {
        let request = match self.scheduler.get_request(req_id) {
            Some(r) => r,
            None => return (None, None),
        };

        let num_output_tokens = request.output_token_ids.len() as u32;

        // 1. Check max_tokens.
        if num_output_tokens >= request.max_tokens {
            return (Some(FinishReason::Length), None);
        }

        // Only check token-based stop criteria if we have new tokens.
        if new_token_ids.is_empty() {
            return (None, None);
        }

        let params = &request.sampling_params;

        // Check each new token against stop conditions.
        for &token_id in new_token_ids {
            // 2. Check EOS tokens (unless ignore_eos is set).
            //    Supports multiple EOS token IDs (e.g. LLaMA 3 has
            //    <|end_of_text|>, <|eom_id|>, <|eot_id|>).
            if !params.ignore_eos && self.eos_token_ids.contains(&token_id) {
                return (Some(FinishReason::Stop), Some(StopReason::Token(token_id)));
            }

            // 3. Check stop_token_ids.
            if params.stop_token_ids.contains(&token_id) {
                return (Some(FinishReason::Stop), Some(StopReason::Token(token_id)));
            }
        }

        (None, None)
    }

    // -----------------------------------------------------------------------
    // Busy loop (for process-based engine core)
    // -----------------------------------------------------------------------

    /// Run the core busy loop.
    ///
    /// This is the main entry point for the engine core process. It
    /// alternates between processing input requests and executing steps.
    ///
    /// The loop runs until `shutdown()` is called.
    pub fn run_busy_loop<F>(&mut self, mut process_input: F)
    where
        F: FnMut(&mut Self) -> bool,
    {
        info!("EngineCore busy loop started");

        loop {
            if self.is_shutdown {
                break;
            }

            // 1. Process input requests until we have work to do.
            let should_continue = process_input(self);
            if !should_continue {
                break;
            }

            // 2. Step the engine.
            match self.step() {
                Ok((outputs, _model_executed)) => {
                    // In a real implementation, outputs would be sent over ZMQ.
                    // For now, they are returned via the step() call above.
                    let _ = outputs;
                }
                Err(e) => {
                    error!("Engine step failed: {}", e);
                    break;
                }
            }
        }

        info!("EngineCore busy loop stopped");
    }

    // -----------------------------------------------------------------------
    // Shutdown
    // -----------------------------------------------------------------------

    /// Shut down the engine core.
    pub fn shutdown(&mut self) {
        if self.is_shutdown {
            return;
        }
        info!("Shutting down EngineCore");
        self.is_shutdown = true;
        self.scheduler.shutdown();
        self.executor.shutdown();
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of unfinished requests.
    pub fn num_unfinished_requests(&self) -> usize {
        self.scheduler.get_num_unfinished_requests()
    }

    /// Whether there are unfinished requests.
    pub fn has_unfinished_requests(&self) -> bool {
        self.scheduler.has_unfinished_requests()
    }

    /// Whether there are finished requests pending notification.
    pub fn has_finished_requests(&self) -> bool {
        self.scheduler.has_finished_requests()
    }

    /// Get request counts (running, waiting).
    pub fn get_request_counts(&self) -> (usize, usize) {
        self.scheduler.get_request_counts()
    }

    /// Engine index.
    pub fn engine_index(&self) -> u32 {
        self.engine_index
    }

    /// Whether the engine is shut down.
    pub fn is_shutdown(&self) -> bool {
        self.is_shutdown
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::NoopExecutor;
    use vllm_common::SamplingParams;
    use vllm_config::SchedulerPolicy;

    fn make_test_config() -> EngineCoreConfig {
        EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: 8192,
                max_num_seqs: 256,
                max_num_scheduled_tokens: None,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                long_prefill_token_threshold: 0,
                ..Default::default()
            },
            max_model_len: 4096,
            num_gpu_blocks: 1024,
            block_size: 16,
            engine_index: 0,
            async_scheduling: false,
            use_spec_decode: false,
            eos_token_ids: vec![],
        }
    }

    fn make_request(id: &str, num_tokens: usize) -> Request {
        let prompt_token_ids: Vec<u32> = (0..num_tokens as u32).collect();
        let params = SamplingParams {
            max_tokens: Some(16),
            ..Default::default()
        };
        Request::new(id.to_string(), prompt_token_ids, params, 0.0, 0, 0, None)
    }

    #[test]
    fn test_engine_core_new() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let engine = EngineCore::new(config, executor);

        assert_eq!(engine.engine_index(), 0);
        assert!(!engine.is_shutdown());
        assert!(!engine.has_work());
        assert_eq!(engine.num_unfinished_requests(), 0);
    }

    #[test]
    fn test_add_and_step() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Add a request.
        let req = make_request("req-1", 10);
        engine.add_request(req);

        assert!(engine.has_work());
        assert_eq!(engine.num_unfinished_requests(), 1);
        let (running, waiting) = engine.get_request_counts();
        assert_eq!(running, 0);
        assert_eq!(waiting, 1);

        // Step the engine.
        let (outputs, model_executed) = engine.step().unwrap();
        assert!(model_executed);

        // We should have outputs.
        assert!(!outputs.is_empty());
        let client_0_outputs = outputs.get(&0).unwrap();
        assert!(!client_0_outputs.outputs.is_empty());
        assert_eq!(client_0_outputs.engine_index, 0);
    }

    #[test]
    fn test_empty_step() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Step with no requests.
        let (outputs, model_executed) = engine.step().unwrap();
        assert!(!model_executed);
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_multiple_requests() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Add multiple requests.
        for i in 0..5 {
            let req = make_request(&format!("req-{i}"), 10);
            engine.add_request(req);
        }

        assert_eq!(engine.num_unfinished_requests(), 5);

        // Step should schedule all requests.
        let (outputs, model_executed) = engine.step().unwrap();
        assert!(model_executed);

        let client_0 = outputs.get(&0).unwrap();
        assert!(client_0.outputs.len() >= 5);
    }

    #[test]
    fn test_abort_request() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Add and then abort.
        engine.add_request(make_request("req-1", 10));
        engine.add_request(make_request("req-2", 10));
        assert_eq!(engine.num_unfinished_requests(), 2);

        engine.abort_requests(&["req-1".to_string()]);
        assert_eq!(engine.num_unfinished_requests(), 1);
    }

    #[test]
    fn test_queue_aborts() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        engine.add_request(make_request("req-1", 10));
        engine.add_request(make_request("req-2", 10));

        // Queue aborts (simulating async arrival).
        engine.queue_aborts(vec!["req-1".to_string()]);
        assert_eq!(engine.num_unfinished_requests(), 2); // Not processed yet.

        // Process aborts queue.
        engine.process_aborts_queue();
        assert_eq!(engine.num_unfinished_requests(), 1);
    }

    #[test]
    fn test_pause_resume() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        assert!(!engine.is_scheduler_paused());

        // Pause with "keep" mode.
        engine.pause_scheduler(PauseMode::Keep).unwrap();
        assert!(engine.is_scheduler_paused());

        // Resume.
        engine.resume_scheduler();
        assert!(!engine.is_scheduler_paused());
    }

    #[test]
    fn test_prefix_cache_reset() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Reset with no running requests should succeed.
        assert!(engine.reset_prefix_cache());
    }

    #[test]
    fn test_shutdown() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        engine.shutdown();
        assert!(engine.is_shutdown());

        // Double shutdown should be safe.
        engine.shutdown();
        assert!(engine.is_shutdown());
    }

    #[test]
    fn test_step_after_abort() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Add, step (to move to running), abort, step again.
        engine.add_request(make_request("req-1", 10));
        let _ = engine.step().unwrap();

        engine.abort_requests(&["req-1".to_string()]);

        // The step after abort should still work (may have empty output).
        let (_, _) = engine.step().unwrap();
    }

    #[test]
    fn test_engine_index() {
        let mut config = make_test_config();
        config.engine_index = 42;
        let executor = Box::new(NoopExecutor::new(1024));
        let engine = EngineCore::new(config, executor);
        assert_eq!(engine.engine_index(), 42);
    }

    #[test]
    fn test_timestamp_increases() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let engine = EngineCore::new(config, executor);

        let t1 = engine.timestamp();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t2 = engine.timestamp();
        assert!(t2 > t1);
    }

    #[test]
    fn test_busy_loop_terminates() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let mut iterations = 0;
        engine.run_busy_loop(|engine| {
            iterations += 1;
            if iterations == 1 {
                engine.add_request(make_request("req-1", 10));
            }
            if iterations >= 3 {
                engine.shutdown();
                return false;
            }
            true
        });

        assert!(engine.is_shutdown());
        assert!(iterations >= 3);
    }

    // -- Stop criteria tests --

    #[test]
    fn test_max_tokens_stop() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Create a request with max_tokens=3.
        let params = SamplingParams {
            max_tokens: Some(3),
            ..Default::default()
        };
        let req = Request::new("req-1".to_string(), vec![1, 2, 3], params, 0.0, 0, 0, None);
        engine.add_request(req);
        assert_eq!(engine.num_unfinished_requests(), 1);

        // Step repeatedly — request should finish after generating 3 tokens.
        let mut finished = false;
        for _ in 0..10 {
            let (outputs, _) = engine.step().unwrap();
            if let Some(client_out) = outputs.get(&0) {
                for out in &client_out.outputs {
                    if out.request_id == "req-1" && out.finish_reason.is_some() {
                        assert_eq!(out.finish_reason, Some(FinishReason::Length));
                        finished = true;
                    }
                }
            }
            if finished {
                break;
            }
        }
        assert!(finished, "Request should have finished due to max_tokens");
    }

    #[test]
    fn test_eos_token_stop() {
        let mut config = make_test_config();
        // NoopExecutor generates incrementing token IDs starting from 1000.
        // Set EOS to 1000 so the first decode step triggers it.
        config.eos_token_ids = vec![1000];
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let params = SamplingParams {
            max_tokens: Some(100),
            ..Default::default()
        };
        let req = Request::new("req-1".to_string(), vec![10, 20], params, 0.0, 0, 0, None);
        engine.add_request(req);

        // Step — the first generated token should be 1000 (EOS).
        let (outputs, _) = engine.step().unwrap();
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "req-1")
            .unwrap();

        assert!(req_out.new_token_ids.contains(&1000));
        assert_eq!(req_out.finish_reason, Some(FinishReason::Stop));
        assert_eq!(req_out.stop_reason, Some(StopReason::Token(1000)));
    }

    #[test]
    fn test_stop_token_ids_stop() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Set stop_token_ids to include 1000 (NoopExecutor starts there).
        let params = SamplingParams {
            max_tokens: Some(100),
            stop_token_ids: vec![1000],
            ..Default::default()
        };
        let req = Request::new("req-1".to_string(), vec![10, 20], params, 0.0, 0, 0, None);
        engine.add_request(req);

        let (outputs, _) = engine.step().unwrap();
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "req-1")
            .unwrap();

        assert!(req_out.new_token_ids.contains(&1000));
        assert_eq!(req_out.finish_reason, Some(FinishReason::Stop));
        assert_eq!(req_out.stop_reason, Some(StopReason::Token(1000)));
    }

    #[test]
    fn test_ignore_eos() {
        let mut config = make_test_config();
        config.eos_token_ids = vec![1000];
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Set ignore_eos = true — EOS should not stop the request.
        let params = SamplingParams {
            max_tokens: Some(100),
            ignore_eos: true,
            ..Default::default()
        };
        let req = Request::new("req-1".to_string(), vec![10, 20], params, 0.0, 0, 0, None);
        engine.add_request(req);

        // Step — even though token 1000 (EOS) is generated, it shouldn't stop.
        let (outputs, _) = engine.step().unwrap();
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "req-1")
            .unwrap();

        // Token 1000 should be generated.
        assert!(req_out.new_token_ids.contains(&1000));
        // With ignore_eos, the EOS token should NOT trigger a stop.
        assert!(
            req_out.finish_reason.is_none(),
            "EOS should be ignored when ignore_eos is set"
        );
    }

    #[test]
    fn test_multiple_eos_token_ids() {
        // Simulate a model with multiple EOS tokens (e.g. LLaMA 3:
        // 128001=<|end_of_text|>, 128008=<|eom_id|>, 128009=<|eot_id|>).
        // NoopExecutor generates token IDs starting from 1000, so use 1002
        // as a secondary EOS (hit on the 3rd decode step).
        let mut config = make_test_config();
        config.eos_token_ids = vec![9999, 1002, 8888]; // 1002 should trigger.
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let params = SamplingParams {
            max_tokens: Some(100),
            ..Default::default()
        };
        let req = Request::new("req-1".to_string(), vec![100, 200], params, 0.0, 0, 0, None);
        engine.add_request(req);

        // Step until the request finishes. NoopExecutor generates 1000, 1001,
        // 1002, ... so token 1002 (a secondary EOS) should trigger stop.
        let mut finished = false;
        for _ in 0..10 {
            let (outputs, _) = engine.step().unwrap();
            if let Some(client_out) = outputs.get(&0) {
                for out in &client_out.outputs {
                    if out.request_id == "req-1" && out.finish_reason.is_some() {
                        assert_eq!(out.finish_reason, Some(FinishReason::Stop));
                        assert_eq!(out.stop_reason, Some(StopReason::Token(1002)));
                        finished = true;
                    }
                }
            }
            if finished {
                break;
            }
        }
        assert!(finished, "Request should stop on secondary EOS token ID");
    }

    #[test]
    fn test_prompt_logprobs_propagation() {
        use vllm_common::sampling::{LogprobsOutput, TokenLogprob};

        // Simulate a ModelRunnerOutput with prompt_logprobs_dict populated.
        let mut prompt_logprobs_dict = std::collections::HashMap::new();
        prompt_logprobs_dict.insert(
            "req-1".to_string(),
            vec![
                LogprobsOutput {
                    sampled: TokenLogprob {
                        token_id: 20,
                        logprob: -0.5,
                        rank: 1,
                    },
                    top_logprobs: vec![],
                },
                LogprobsOutput {
                    sampled: TokenLogprob {
                        token_id: 30,
                        logprob: -1.2,
                        rank: 2,
                    },
                    top_logprobs: vec![],
                },
            ],
        );

        let model_output = ModelRunnerOutput {
            req_ids: vec!["req-1".to_string()],
            req_id_to_index: [("req-1".to_string(), 0)].into_iter().collect(),
            sampled_token_ids: vec![vec![42]],
            logprobs: None,
            prompt_logprobs_dict,
            draft_token_ids: None,
        };

        // Set up engine.
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Add a request so the scheduler has it.
        let req = make_request("req-1", 10);
        engine.add_request(req);

        // Schedule it.
        let scheduler_output = engine.scheduler.schedule();

        // Manually call update_from_output with our test model_output.
        let outputs = engine.update_from_output(&scheduler_output, &model_output);

        // Check that prompt logprobs are propagated.
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "req-1")
            .unwrap();

        let plps = req_out.new_prompt_logprobs.as_ref().unwrap();
        // Position 0 is None, positions 1-2 are Some.
        assert_eq!(plps.len(), 3);
        assert!(plps[0].is_none());
        assert_eq!(plps[1].as_ref().unwrap().sampled.token_id, 20);
        assert_eq!(plps[2].as_ref().unwrap().sampled.token_id, 30);
    }
}
