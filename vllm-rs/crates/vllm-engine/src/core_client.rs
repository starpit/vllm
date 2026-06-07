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

use std::collections::HashMap;

use tracing::info;
use vllm_common::{EngineCoreOutputs, EngineCoreRequest, Request};
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_protocol::messages::PauseMode;

use crate::engine_core::{EngineCore, EngineCoreConfig, StepOutputs};
use crate::error::{EngineError, EngineResult};
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
    ///
    /// Returns `(outputs, model_executed)` where `model_executed` indicates
    /// whether the model actually ran a forward pass this step.
    fn get_output(&mut self) -> EngineResult<(EngineCoreOutputs, bool)>;

    /// Add a new inference request.
    fn add_request(&mut self, request: EngineCoreRequest) -> EngineResult<()>;

    /// Abort requests by ID.
    fn abort_requests(&mut self, request_ids: &[String]) -> EngineResult<()>;

    /// Abort all currently running requests.
    ///
    /// Used when a persistent executor error (OOM, shape mismatch) makes it
    /// impossible to continue processing the current batch. Drains the
    /// scheduler's running queue and frees their KV blocks.
    fn abort_running_requests(&mut self);

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

    /// Put the engine to sleep, freeing GPU memory.
    fn sleep(&mut self, level: u32) -> EngineResult<()>;

    /// Wake the engine from sleep.
    fn wake_up(&mut self, tags: Option<&[String]>) -> EngineResult<()>;

    /// Whether the engine is currently sleeping.
    fn is_sleeping(&self) -> bool;

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
/// Directly calls methods on an `EngineCore` instance. When async scheduling
/// is enabled, spawns a background executor thread with a 2-batch pipeline
/// to overlap CPU scheduling with GPU execution (matching Python's
/// `EngineCore.step_with_batch_queue()`).
///
/// Port of: `vllm/v1/engine/core_client.py::InprocClient`
pub struct InprocClient {
    engine: EngineCore,
    /// Pipeline state for async scheduling. `None` when async scheduling is
    /// disabled (falls back to synchronous `step()`).
    pipeline: Option<PipelineState>,
}

/// Background pipeline state for async scheduling.
///
/// Three OS threads — engine (main), executor, output — connected by typed
/// `crossbeam_channel` channels. Mirrors python's `step_with_batch_queue` +
/// `WorkerAsyncOutput` ThreadPoolExecutor shape, but uses real ownership /
/// drop semantics instead of python's GIL-shaped scaffolding.
///
/// Lifecycle:
/// - `engine` produces `SchedulerOutput` and `try_send`s it on `sched_tx`.
/// - `executor` runs forward+sample, ships `(sched, output)` on `unresolved_tx`.
///   Output is "deferred" — its sampled tokens still on the GPU behind a CUDA
///   event, the resolve closure not yet called.
/// - `output` thread calls `output.resolve()` (the cuEventSynchronize and
///   pinned-buffer read), then ships the resolved pair on `resolved_tx`.
/// - `engine` recv's from `resolved_rx` and runs `finalize_step`. The
///   cuEventSynchronize is now off the engine thread — that's the perf win.
///
/// In-flight tracking is the bounded channels themselves; no `gpu_in_flight`
/// counter, no `deferred: Option<...>` field. `Box<SchedulerOutput>` is gone
/// — `SchedulerOutput` moves by value through the typed channels.
struct PipelineState {
    sched_tx: std::sync::mpsc::SyncSender<PipelineMsg>,
    resolved_rx: std::sync::mpsc::Receiver<PipelineResult>,
    /// Outstanding sched_tx sends not yet matched by a resolved_rx recv.
    /// Tracked here only because `mpsc::SyncSender::len()` doesn't exist;
    /// crossbeam_channel would let us drop this field.
    in_flight: u32,
    _executor_thread: std::thread::JoinHandle<()>,
    _output_thread: std::thread::JoinHandle<()>,
}

/// A cloneable, Send+Sync handle for computing embeddings via the
/// background executor pipeline. Obtained from [`InprocClient::embed_sender`].
#[derive(Clone)]
pub struct EmbedSender {
    tx: std::sync::mpsc::SyncSender<PipelineMsg>,
}

impl EmbedSender {
    /// Compute embeddings for the given token ID sequences.
    /// Blocks until the executor thread returns the result.
    pub fn embed(&self, token_id_seqs: Vec<Vec<u32>>) -> EngineResult<Vec<Vec<f32>>> {
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
        self.tx
            .send(PipelineMsg::Embed(token_id_seqs, reply_tx))
            .map_err(|_| EngineError::Executor("executor thread exited".into()))?;
        reply_rx
            .recv()
            .map_err(|_| EngineError::Executor("executor thread exited".into()))?
    }
}

/// Engine → executor channel payload.
///
/// `SchedulerOutput` is intentionally moved by value (no `Box`) to mirror
/// python's pipeline shape, so clippy's `large-enum-variant` lint is muted
/// here. The size delta (~480 bytes vs the Embed variant's ~40 bytes) is
/// negligible at our channel depth (2) — boxing would add a heap alloc per
/// step on the hot path for no real gain.
#[allow(clippy::large_enum_variant)]
enum PipelineMsg {
    /// Execute a model step.
    Step(SchedulerOutput),
    /// Compute embeddings (bypasses scheduler).
    Embed(
        Vec<Vec<u32>>,
        std::sync::mpsc::SyncSender<EngineResult<Vec<Vec<f32>>>>,
    ),
}

/// Executor → output channel payload. The `ModelRunnerOutput` may be
/// deferred (`d2h_resolver: Some(...)`); the output thread resolves it.
enum UnresolvedMsg {
    Step(SchedulerOutput, EngineResult<ModelRunnerOutput>),
}

/// Output → engine channel payload. By the time the engine recv's this,
/// `ModelRunnerOutput::sampled_token_ids` is host-resident — the
/// cuEventSynchronize already happened on the output thread.
enum PipelineResult {
    Step(SchedulerOutput, EngineResult<ModelRunnerOutput>),
}

/// Executor thread: pull `PipelineMsg`, run `execute_model` (which may
/// return a deferred output whose tokens are still on the GPU), ship the
/// pair to the output thread for resolution.
///
/// Embed requests get short-circuited back to the caller without touching
/// the output thread (they're synchronous and not on the perf-critical
/// hot path).
fn executor_bg_loop(
    mut executor: Box<dyn Executor>,
    sched_rx: std::sync::mpsc::Receiver<PipelineMsg>,
    unresolved_tx: std::sync::mpsc::SyncSender<UnresolvedMsg>,
) {
    while let Ok(msg) = sched_rx.recv() {
        match msg {
            PipelineMsg::Step(sched) => {
                let result = executor.execute_model(&sched);
                if unresolved_tx
                    .send(UnresolvedMsg::Step(sched, result))
                    .is_err()
                {
                    break;
                }
            }
            PipelineMsg::Embed(token_id_seqs, reply) => {
                let _ = reply.send(executor.embed(token_id_seqs));
            }
        }
    }
    executor.shutdown();
}

/// Output thread: pull deferred `(sched, output)` pairs, resolve the D2H
/// (cuEventSynchronize + pinned-buffer read), ship the resolved pair to
/// the engine.
///
/// This is the work that python's `WorkerAsyncOutput` ThreadPoolExecutor
/// does — the resolve sits between executor and engine so neither blocks
/// on the GPU sync.
fn output_bg_loop(
    unresolved_rx: std::sync::mpsc::Receiver<UnresolvedMsg>,
    resolved_tx: std::sync::mpsc::SyncSender<PipelineResult>,
) {
    while let Ok(UnresolvedMsg::Step(sched, result)) = unresolved_rx.recv() {
        let resolved = result.map(|mut o| {
            o.resolve();
            o
        });
        if resolved_tx
            .send(PipelineResult::Step(sched, resolved))
            .is_err()
        {
            break;
        }
    }
}

impl InprocClient {
    /// Create a new in-process client, initializing the engine core.
    ///
    /// When `async_scheduling` is enabled, the pipeline thread is started
    /// lazily on the first `get_output()` call. This allows the server path
    /// to call `take_executor()` first for its own background thread.
    pub fn new(config: EngineCoreConfig, executor: Box<dyn Executor>) -> Self {
        Self {
            engine: EngineCore::new(config, executor),
            pipeline: None,
        }
    }

    /// Start the background executor pipeline for overlapping CPU scheduling
    /// with GPU execution. Called explicitly by the LLM path.
    ///
    /// Does nothing if the pipeline is already started or the executor has
    /// been taken by the server path.
    pub fn start_pipeline(&mut self) {
        if self.pipeline.is_some() {
            return;
        }
        if let Some(executor) = self.engine.take_executor() {
            info!(
                "InprocClient: spawning vllm-executor + vllm-output threads for pipelined execution"
            );
            // Three channels at depth 2 (matches python's max_concurrent_batches=2).
            let (sched_tx, sched_rx) = std::sync::mpsc::sync_channel::<PipelineMsg>(2);
            let (unresolved_tx, unresolved_rx) = std::sync::mpsc::sync_channel::<UnresolvedMsg>(2);
            let (resolved_tx, resolved_rx) = std::sync::mpsc::sync_channel::<PipelineResult>(2);
            let executor_thread = std::thread::Builder::new()
                .name("vllm-executor".into())
                .spawn(move || executor_bg_loop(executor, sched_rx, unresolved_tx))
                .expect("failed to spawn executor thread");
            let output_thread = std::thread::Builder::new()
                .name("vllm-output".into())
                .spawn(move || output_bg_loop(unresolved_rx, resolved_tx))
                .expect("failed to spawn output thread");
            self.pipeline = Some(PipelineState {
                sched_tx,
                resolved_rx,
                in_flight: 0,
                _executor_thread: executor_thread,
                _output_thread: output_thread,
            });
        }
    }

    /// Get an [`EmbedSender`] for computing embeddings via the pipeline.
    /// Returns `None` if the pipeline has not been started.
    pub fn embed_sender(&self) -> Option<EmbedSender> {
        self.pipeline.as_ref().map(|p| EmbedSender {
            tx: p.sched_tx.clone(),
        })
    }

    /// Get a reference to the inner engine core.
    pub fn engine(&self) -> &EngineCore {
        &self.engine
    }

    /// Whether there are unfinished requests (including pipelined in-flight).
    pub fn has_unfinished_requests(&self) -> bool {
        if self.engine.has_unfinished_requests() {
            return true;
        }
        if let Some(ref pipeline) = self.pipeline
            && pipeline.in_flight > 0
        {
            return true;
        }
        false
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
        request.block_annotations = ec_request.block_annotations;
        request.seal = ec_request.seal;
        request.volatile = ec_request.volatile;

        self.engine.add_request(request);
        Ok(())
    }

    /// Pipelined get_output: 3-thread shape, cuEventSynchronize off the
    /// engine thread.
    ///
    /// 1. Pre-schedule: fill `sched_tx` up to its bound (2 in flight).
    /// 2. Block on `resolved_rx` — recv'd output already has its D2H
    ///    sync done by the output thread, so this is a pure cross-thread
    ///    wait, not a CUDA sync.
    /// 3. Run `finalize_step` on the engine thread (mutates scheduler state).
    fn get_output_pipelined(
        engine: &mut EngineCore,
        pipeline: &mut PipelineState,
    ) -> EngineResult<(StepOutputs, bool)> {
        // 1. Pre-schedule: keep the executor and output threads fed.
        //    `try_send` returns Full when sched_tx is at capacity, in which
        //    case we hand the SchedulerOutput back to the scheduler queue.
        //    For the bench shape (decode-only, no chunked-prefill resume)
        //    this loop terminates on the first `schedule_next() -> None`.
        while pipeline.in_flight < 2 {
            let Some(sched) = engine.schedule_next() else {
                break;
            };
            match pipeline.sched_tx.try_send(PipelineMsg::Step(sched)) {
                Ok(()) => pipeline.in_flight += 1,
                Err(std::sync::mpsc::TrySendError::Full(_)) => break,
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    return Err(EngineError::Executor("executor thread exited".into()));
                }
            }
        }

        // 2. Nothing to drain → empty outputs, no model executed.
        if pipeline.in_flight == 0 {
            return Ok((HashMap::new(), false));
        }

        // 3. Block on the oldest resolved output. The cuEventSynchronize
        //    already happened on the output thread, so this is a pure
        //    cross-thread wait.
        let PipelineResult::Step(sched, result) = pipeline
            .resolved_rx
            .recv()
            .map_err(|_| EngineError::Executor("output thread exited".into()))?;
        pipeline.in_flight -= 1;
        let model_output = result?;

        // 4. Finalize on the engine thread (scheduler state mutation).
        let outputs = engine.finalize_step(&sched, &model_output);
        Ok((outputs, true))
    }
}

impl EngineCoreClient for InprocClient {
    fn get_output(&mut self) -> EngineResult<(EngineCoreOutputs, bool)> {
        let (outputs, model_executed) = if let Some(ref mut pipeline) = self.pipeline {
            // Pipelined path: overlap CPU scheduling with GPU execution.
            Self::get_output_pipelined(&mut self.engine, pipeline)?
        } else {
            // Synchronous path.
            self.engine.step()?
        };

        // Merge all client outputs into a single EngineCoreOutputs.
        if outputs.is_empty() {
            return Ok((EngineCoreOutputs::default(), model_executed));
        }
        if let Some(out) = outputs.into_values().next() {
            Ok((out, model_executed))
        } else {
            Ok((EngineCoreOutputs::default(), model_executed))
        }
    }

    fn add_request(&mut self, request: EngineCoreRequest) -> EngineResult<()> {
        self.convert_and_add(request)
    }

    fn abort_requests(&mut self, request_ids: &[String]) -> EngineResult<()> {
        self.engine.abort_requests(request_ids);
        Ok(())
    }

    fn abort_running_requests(&mut self) {
        self.engine.abort_running_requests();
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

    fn sleep(&mut self, level: u32) -> EngineResult<()> {
        self.engine.sleep(level)
    }

    fn wake_up(&mut self, tags: Option<&[String]>) -> EngineResult<()> {
        self.engine.wake_up(tags)
    }

    fn is_sleeping(&self) -> bool {
        self.engine.is_sleeping()
    }

    fn embed(&mut self, token_id_seqs: Vec<Vec<u32>>) -> EngineResult<Vec<Vec<f32>>> {
        if let Some(ref pipeline) = self.pipeline {
            // Route through the background executor thread.
            let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
            pipeline
                .sched_tx
                .send(PipelineMsg::Embed(token_id_seqs, reply_tx))
                .map_err(|_| EngineError::Executor("executor thread exited".into()))?;
            reply_rx
                .recv()
                .map_err(|_| EngineError::Executor("executor thread exited".into()))?
        } else {
            self.engine.embed(token_id_seqs)
        }
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
            proposer_config: None,
            eos_token_ids: vec![],
            is_pooling: false,
            enable_prefix_caching: false,
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
            block_annotations: None,
            seal: false,
            volatile: false,
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
        let (outputs, model_executed) = client.get_output().unwrap();
        assert!(!outputs.outputs.is_empty());
        assert!(model_executed);
    }

    #[test]
    fn test_inproc_empty_output() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client = InprocClient::new(config, executor);

        // No requests, should return empty outputs.
        let (outputs, model_executed) = client.get_output().unwrap();
        assert!(outputs.outputs.is_empty());
        assert!(!model_executed);
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
        let (outputs, _) = client.get_output().unwrap();
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

        let (outputs, _) = client.get_output().unwrap();
        assert!(outputs.outputs.len() >= 10);
    }

    #[test]
    fn test_trait_object() {
        // Ensure InprocClient can be used as a trait object.
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut client: Box<dyn EngineCoreClient> = Box::new(InprocClient::new(config, executor));

        client.add_request(make_ec_request("req-1", 10)).unwrap();
        let (outputs, _) = client.get_output().unwrap();
        assert!(!outputs.outputs.is_empty());
        client.shutdown().unwrap();
    }
}
