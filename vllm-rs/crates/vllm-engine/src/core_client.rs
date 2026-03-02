// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Engine core client: the interface between the API server and the engine
//! core.
//!
//! The client abstracts over different communication modes:
//! - **In-process**: Direct function calls to `EngineCore` (no IPC).
//! - **Multi-process**: ZMQ sockets to a background `EngineCore` process.
//!
//! Port of: `vllm/v1/engine/core_client.py`

use vllm_common::{EngineCoreOutputs, EngineCoreRequest, Request};
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_protocol::messages::PauseMode;

use crate::engine_core::{EngineCore, EngineCoreConfig, StepOutputs};
use crate::error::EngineResult;
use crate::executor::{Executor, ModelRunnerOutput};

// ---------------------------------------------------------------------------
// EngineCoreClient trait
// ---------------------------------------------------------------------------

/// The interface between the API server front-end and the engine core.
///
/// Subclasses handle different communication modes (in-process, ZMQ, etc.).
///
/// Port of: `vllm/v1/engine/core_client.py::EngineCoreClient`
pub trait EngineCoreClient {
    /// Get the output from the latest engine step.
    ///
    /// For the in-process client, this runs a step and returns the outputs.
    /// For the multi-process client, this receives outputs from ZMQ.
    fn get_output(&mut self) -> EngineResult<EngineCoreOutputs>;

    /// Add a new inference request.
    fn add_request(&mut self, request: EngineCoreRequest) -> EngineResult<()>;

    /// Abort requests by ID.
    fn abort_requests(&mut self, request_ids: &[String]) -> EngineResult<()>;

    /// Shut down the engine.
    fn shutdown(&mut self) -> EngineResult<()>;

    /// Reset the prefix cache.
    fn reset_prefix_cache(&mut self) -> EngineResult<bool>;

    /// Pause the scheduler.
    fn pause_scheduler(&mut self, mode: PauseMode) -> EngineResult<()>;

    /// Resume the scheduler.
    fn resume_scheduler(&mut self) -> EngineResult<()>;

    /// Whether the scheduler is paused.
    fn is_scheduler_paused(&self) -> bool;

    /// Compute embeddings for the given token ID sequences.
    ///
    /// Bypasses the scheduler — embedding is a single prefill pass with no
    /// KV cache or decode loop.
    fn embed(&mut self, _token_id_seqs: Vec<Vec<u32>>) -> EngineResult<Vec<Vec<f32>>> {
        Err(crate::error::EngineError::Executor(
            "embedding not supported".into(),
        ))
    }

    // -------------------------------------------------------------------
    // Async scheduling split ops (default: unsupported)
    // -------------------------------------------------------------------

    /// Take the executor out for use on a dedicated thread.
    fn take_executor(&mut self) -> Option<Box<dyn Executor>> {
        None
    }

    /// Post-execution processing: update state from model output.
    fn finalize_step(
        &mut self,
        _sched: &SchedulerOutput,
        _model: &ModelRunnerOutput,
    ) -> EngineResult<StepOutputs> {
        Err(crate::error::EngineError::Executor(
            "finalize_step not supported".into(),
        ))
    }

    /// Run scheduling if there is work to do.
    fn schedule_next(&mut self) -> EngineResult<Option<SchedulerOutput>> {
        Err(crate::error::EngineError::Executor(
            "schedule_next not supported".into(),
        ))
    }

    /// Whether async scheduling is enabled on the underlying engine.
    fn async_scheduling(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// InprocClient
// ---------------------------------------------------------------------------

/// In-process engine core client.
///
/// Directly calls methods on an `EngineCore` instance. Used for V0-style
/// synchronous `add_request()` + `step()` usage patterns.
///
/// Port of: `vllm/v1/engine/core_client.py::InprocClient`
pub struct InprocClient {
    engine: EngineCore,
}

impl InprocClient {
    /// Create a new in-process client, initializing the engine core.
    pub fn new(config: EngineCoreConfig, executor: Box<dyn Executor>) -> Self {
        Self {
            engine: EngineCore::new(config, executor),
        }
    }

    /// Get a reference to the inner engine core.
    pub fn engine(&self) -> &EngineCore {
        &self.engine
    }

    /// Get a mutable reference to the inner engine core.
    pub fn engine_mut(&mut self) -> &mut EngineCore {
        &mut self.engine
    }

    /// Convert an `EngineCoreRequest` to a `Request` and add it.
    fn convert_and_add(&mut self, ec_request: EngineCoreRequest) -> EngineResult<()> {
        let params = ec_request.sampling_params.unwrap_or_default();
        let prompt_token_ids = ec_request.prompt_token_ids.unwrap_or_default();

        let mut request = Request::new(
            ec_request.request_id,
            prompt_token_ids,
            params,
            ec_request.arrival_time,
            ec_request.client_index,
            ec_request.priority,
            ec_request.cache_salt,
        );
        request.is_pooling = ec_request.is_pooling;
        request.mm_data = ec_request.mm_data;

        self.engine.add_request(request);
        Ok(())
    }
}

impl EngineCoreClient for InprocClient {
    fn get_output(&mut self) -> EngineResult<EngineCoreOutputs> {
        let (outputs, _model_executed) = self.engine.step()?;

        // Merge all client outputs into a single EngineCoreOutputs.
        // In the in-process case, there's typically just one client (index 0).
        if outputs.is_empty() {
            return Ok(EngineCoreOutputs::default());
        }

        // Return client 0's outputs, or merge all clients.
        if let Some(out) = outputs.into_values().next() {
            Ok(out)
        } else {
            Ok(EngineCoreOutputs::default())
        }
    }

    fn add_request(&mut self, request: EngineCoreRequest) -> EngineResult<()> {
        self.convert_and_add(request)
    }

    fn abort_requests(&mut self, request_ids: &[String]) -> EngineResult<()> {
        self.engine.abort_requests(request_ids);
        Ok(())
    }

    fn shutdown(&mut self) -> EngineResult<()> {
        self.engine.shutdown();
        Ok(())
    }

    fn reset_prefix_cache(&mut self) -> EngineResult<bool> {
        Ok(self.engine.reset_prefix_cache())
    }

    fn pause_scheduler(&mut self, mode: PauseMode) -> EngineResult<()> {
        self.engine.pause_scheduler(mode)?;
        Ok(())
    }

    fn resume_scheduler(&mut self) -> EngineResult<()> {
        self.engine.resume_scheduler();
        Ok(())
    }

    fn is_scheduler_paused(&self) -> bool {
        self.engine.is_scheduler_paused()
    }

    fn embed(&mut self, token_id_seqs: Vec<Vec<u32>>) -> EngineResult<Vec<Vec<f32>>> {
        self.engine.embed(token_id_seqs)
    }

    fn take_executor(&mut self) -> Option<Box<dyn Executor>> {
        self.engine.take_executor()
    }

    fn finalize_step(
        &mut self,
        sched: &SchedulerOutput,
        model: &ModelRunnerOutput,
    ) -> EngineResult<StepOutputs> {
        Ok(self.engine.finalize_step(sched, model))
    }

    fn schedule_next(&mut self) -> EngineResult<Option<SchedulerOutput>> {
        Ok(self.engine.schedule_next())
    }

    fn async_scheduling(&self) -> bool {
        self.engine.async_scheduling()
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
    use vllm_config::{SchedulerConfig, SchedulerPolicy};

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
            ngram_proposer_config: None,
            eos_token_ids: vec![],
            is_pooling: false,
        }
    }

    fn make_ec_request(id: &str, num_tokens: usize) -> EngineCoreRequest {
        EngineCoreRequest {
            request_id: id.to_string(),
            prompt_token_ids: Some((0..num_tokens as u32).collect()),
            sampling_params: Some(SamplingParams {
                max_tokens: Some(16),
                ..Default::default()
            }),
            arrival_time: 0.0,
            client_index: 0,
            priority: 0,
            cache_salt: None,
            data_parallel_rank: None,
            is_pooling: false,
            mm_data: None,
        }
    }

    #[test]
    fn test_inproc_client_new() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let client = InprocClient::new(config, executor);
        assert!(!client.engine().is_shutdown());
    }

    #[test]
    fn test_inproc_add_and_get_output() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        // Add a request.
        client.add_request(make_ec_request("req-1", 10)).unwrap();

        // Get output (triggers a step).
        let outputs = client.get_output().unwrap();
        assert!(!outputs.outputs.is_empty());
    }

    #[test]
    fn test_inproc_empty_output() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        // No requests, should return empty outputs.
        let outputs = client.get_output().unwrap();
        assert!(outputs.outputs.is_empty());
    }

    #[test]
    fn test_inproc_abort() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        client.add_request(make_ec_request("req-1", 10)).unwrap();
        client.add_request(make_ec_request("req-2", 10)).unwrap();

        client.abort_requests(&["req-1".to_string()]).unwrap();

        assert_eq!(client.engine().num_unfinished_requests(), 1);
    }

    #[test]
    fn test_inproc_pause_resume() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        assert!(!client.is_scheduler_paused());

        client.pause_scheduler(PauseMode::Keep).unwrap();
        assert!(client.is_scheduler_paused());

        client.resume_scheduler().unwrap();
        assert!(!client.is_scheduler_paused());
    }

    #[test]
    fn test_inproc_reset_prefix_cache() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        assert!(client.reset_prefix_cache().unwrap());
    }

    #[test]
    fn test_inproc_shutdown() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        client.shutdown().unwrap();
        assert!(client.engine().is_shutdown());
    }

    #[test]
    fn test_inproc_multiple_steps() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        client.add_request(make_ec_request("req-1", 10)).unwrap();

        // First step should produce output.
        let outputs = client.get_output().unwrap();
        assert!(!outputs.outputs.is_empty());

        // Subsequent steps should not error (request may or may not produce
        // output depending on scheduler state).
        for _ in 0..4 {
            let _ = client.get_output().unwrap();
        }
    }

    #[test]
    fn test_inproc_multiple_requests() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        for i in 0..10 {
            client
                .add_request(make_ec_request(&format!("req-{i}"), 10))
                .unwrap();
        }

        let outputs = client.get_output().unwrap();
        assert!(outputs.outputs.len() >= 10);
    }

    #[test]
    fn test_trait_object() {
        // Ensure InprocClient can be used as a trait object.
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client: Box<dyn EngineCoreClient> = Box::new(InprocClient::new(config, executor));

        client.add_request(make_ec_request("req-1", 10)).unwrap();
        let outputs = client.get_output().unwrap();
        assert!(!outputs.outputs.is_empty());
        client.shutdown().unwrap();
    }
}
