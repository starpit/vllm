// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Multi-process executor.
//!
//! The `MultiprocExecutor` manages multiple worker processes, one per GPU.
//! It communicates with workers via channels and coordinates collective
//! operations like model execution and KV cache initialization.
//!
//! Port of: `vllm/v1/executor/multiproc_executor.py::MultiprocExecutor`
//!
//! # Architecture
//!
//! ```text
//!   MultiprocExecutor
//!     ├── WorkerHandle (rank 0) ──► worker process/task
//!     ├── WorkerHandle (rank 1) ──► worker process/task
//!     └── WorkerHandle (rank N) ──► worker process/task
//! ```
//!
//! Each worker handle communicates with its worker via tokio channels.
//! The executor broadcasts scheduler outputs to all workers and collects
//! responses from the output rank (TP rank 0 of the last PP stage).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_engine::error::EngineResult;
use vllm_engine::executor::{Executor, ModelRunnerOutput};

use crate::error::{ExecutorError, ExecutorResult};
use crate::parallel::ResolvedParallelConfig;
use crate::worker::Worker;

// ---------------------------------------------------------------------------
// Worker RPC message types
// ---------------------------------------------------------------------------

/// A request sent from the executor to a worker.
#[derive(Debug)]
pub enum WorkerRequest {
    /// Execute the model with the given scheduler output.
    ExecuteModel(Box<SchedulerOutput>),
    /// Initialize KV cache.
    InitializeCache {
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    },
    /// Determine available memory.
    DetermineAvailableMemory,
    /// Check health.
    CheckHealth,
    /// Sleep the worker.
    Sleep(u32),
    /// Wake up the worker.
    WakeUp(Option<Vec<String>>),
    /// Shut down the worker.
    Shutdown,
}

/// A response from a worker to the executor.
#[derive(Debug)]
pub enum WorkerResponse {
    /// Model execution completed.
    ModelOutput(ExecutorResult<ModelRunnerOutput>),
    /// Cache initialization completed.
    CacheInitialized(ExecutorResult<()>),
    /// Available memory reported.
    AvailableMemory(ExecutorResult<usize>),
    /// Health check completed.
    HealthOk(ExecutorResult<()>),
    /// Sleep completed.
    SleepDone(ExecutorResult<()>),
    /// Wake up completed.
    WakeUpDone(ExecutorResult<()>),
    /// Shutdown acknowledged.
    ShutdownAck,
}

// ---------------------------------------------------------------------------
// WorkerHandle
// ---------------------------------------------------------------------------

/// Handle for communicating with a worker.
///
/// Holds the sending end of the request channel and metadata about the worker.
pub struct WorkerHandle {
    /// The worker's global rank.
    pub rank: usize,
    /// Channel to send requests to the worker.
    request_tx: mpsc::UnboundedSender<(WorkerRequest, oneshot::Sender<WorkerResponse>)>,
    /// Whether this worker is the output rank.
    pub is_output_rank: bool,
}

impl WorkerHandle {
    /// Send a request to the worker and wait for a response.
    async fn rpc(&self, request: WorkerRequest) -> ExecutorResult<WorkerResponse> {
        let (response_tx, response_rx) = oneshot::channel();
        self.request_tx
            .send((request, response_tx))
            .map_err(|_| ExecutorError::WorkerDied { rank: self.rank })?;
        response_rx
            .await
            .map_err(|_| ExecutorError::WorkerDied { rank: self.rank })
    }
}

// ---------------------------------------------------------------------------
// Worker task
// ---------------------------------------------------------------------------

/// Run a worker's RPC loop.
///
/// This is spawned as a tokio task. It receives requests from the executor,
/// dispatches them to the worker, and sends back responses.
async fn worker_task(
    mut worker: Box<dyn Worker>,
    mut request_rx: mpsc::UnboundedReceiver<(WorkerRequest, oneshot::Sender<WorkerResponse>)>,
    is_failed: Arc<AtomicBool>,
) {
    while let Some((request, response_tx)) = request_rx.recv().await {
        let response = match request {
            WorkerRequest::ExecuteModel(sched_output) => {
                WorkerResponse::ModelOutput(worker.execute_model(&sched_output))
            }
            WorkerRequest::InitializeCache {
                num_gpu_blocks,
                num_cpu_blocks,
            } => WorkerResponse::CacheInitialized(
                worker.initialize_cache(num_gpu_blocks, num_cpu_blocks),
            ),
            WorkerRequest::DetermineAvailableMemory => {
                WorkerResponse::AvailableMemory(worker.determine_available_memory())
            }
            WorkerRequest::CheckHealth => WorkerResponse::HealthOk(worker.check_health()),
            WorkerRequest::Sleep(level) => WorkerResponse::SleepDone(worker.sleep(level)),
            WorkerRequest::WakeUp(tags) => {
                WorkerResponse::WakeUpDone(worker.wake_up(tags.as_deref()))
            }
            WorkerRequest::Shutdown => {
                worker.shutdown();
                let _ = response_tx.send(WorkerResponse::ShutdownAck);
                return; // Exit the task loop.
            }
        };

        if response_tx.send(response).is_err() {
            // Executor dropped the response channel.
            warn!(
                "Worker rank {} could not send response, executor may have shut down",
                worker.rank()
            );
            is_failed.store(true, Ordering::Relaxed);
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// MultiprocExecutor
// ---------------------------------------------------------------------------

/// Multi-process executor that manages multiple workers.
///
/// Workers run as separate tokio tasks (in-process concurrency). For true
/// process isolation, workers would be spawned as separate OS processes
/// communicating via shared memory or sockets (future work).
///
/// Port of: `vllm/v1/executor/multiproc_executor.py::MultiprocExecutor`
pub struct MultiprocExecutor {
    /// Worker handles, indexed by rank.
    workers: Vec<WorkerHandle>,
    /// Parallel configuration.
    parallel_config: ResolvedParallelConfig,
    /// Whether the executor is sleeping.
    is_sleeping: bool,
    /// Whether the executor has failed.
    is_failed: Arc<AtomicBool>,
    /// Whether the executor has been shut down.
    is_shutdown: bool,
    /// Tokio runtime handle for blocking on async operations.
    runtime: tokio::runtime::Handle,
}

impl MultiprocExecutor {
    /// Create a new multi-process executor from a list of workers.
    ///
    /// Workers must be provided in rank order. Each worker is spawned as a
    /// separate tokio task.
    ///
    /// The `runtime` handle is used to block on async operations from
    /// synchronous `Executor` trait methods.
    pub fn new(
        workers: Vec<Box<dyn Worker>>,
        parallel_config: ResolvedParallelConfig,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        let is_failed = Arc::new(AtomicBool::new(false));
        let output_rank = parallel_config.output_rank();

        let handles: Vec<WorkerHandle> = workers
            .into_iter()
            .enumerate()
            .map(|(rank, worker)| {
                let (request_tx, request_rx) = mpsc::unbounded_channel();
                let failed = Arc::clone(&is_failed);

                // Spawn worker task.
                runtime.spawn(worker_task(worker, request_rx, failed));

                WorkerHandle {
                    rank,
                    request_tx,
                    is_output_rank: rank == output_rank,
                }
            })
            .collect();

        info!(
            "MultiprocExecutor: spawned {} workers, output_rank={}",
            handles.len(),
            output_rank
        );

        Self {
            workers: handles,
            parallel_config,
            is_sleeping: false,
            is_failed,
            is_shutdown: false,
            runtime,
        }
    }

    /// Check if the executor has failed.
    pub fn is_failed(&self) -> bool {
        self.is_failed.load(Ordering::Relaxed)
    }

    /// Number of workers.
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Broadcast a request to all workers and collect responses.
    fn collective_rpc_blocking(
        &self,
        make_request: impl Fn() -> WorkerRequest,
    ) -> ExecutorResult<Vec<WorkerResponse>> {
        if self.is_failed() {
            return Err(ExecutorError::Communication(
                "executor has failed".to_string(),
            ));
        }

        tokio::task::block_in_place(|| {
            self.runtime.block_on(async {
                let mut response_futures = Vec::with_capacity(self.workers.len());
                for handle in &self.workers {
                    response_futures.push(handle.rpc(make_request()));
                }

                let mut responses = Vec::with_capacity(response_futures.len());
                for future in response_futures {
                    responses.push(future.await?);
                }
                Ok(responses)
            })
        })
    }

    /// Send a request to the output rank only and get the response.
    #[allow(dead_code)]
    fn rpc_output_rank_blocking(&self, request: WorkerRequest) -> ExecutorResult<WorkerResponse> {
        if self.is_failed() {
            return Err(ExecutorError::Communication(
                "executor has failed".to_string(),
            ));
        }

        let output_handle = self
            .workers
            .iter()
            .find(|w| w.is_output_rank)
            .ok_or_else(|| ExecutorError::Config("no output rank worker found".to_string()))?;

        tokio::task::block_in_place(|| self.runtime.block_on(output_handle.rpc(request)))
    }
}

impl Executor for MultiprocExecutor {
    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> EngineResult<ModelRunnerOutput> {
        // Broadcast to all workers, collect from output rank.
        let responses = self
            .collective_rpc_blocking(|| {
                WorkerRequest::ExecuteModel(Box::new(scheduler_output.clone()))
            })
            .map_err(|e| vllm_engine::error::EngineError::Executor(e.to_string()))?;

        // Only the output rank's response matters.
        let output_response = responses
            .into_iter()
            .enumerate()
            .find(|(i, _)| self.workers[*i].is_output_rank)
            .map(|(_, r)| r)
            .ok_or_else(|| {
                vllm_engine::error::EngineError::Executor("no output rank response".to_string())
            })?;

        match output_response {
            WorkerResponse::ModelOutput(result) => {
                result.map_err(|e| vllm_engine::error::EngineError::Executor(e.to_string()))
            }
            other => Err(vllm_engine::error::EngineError::Executor(format!(
                "unexpected response: {other:?}"
            ))),
        }
    }

    fn max_concurrent_batches(&self) -> usize {
        // PP requires PP-size concurrent batches to fill the pipeline.
        let pp_size = self.parallel_config.pp_group.world_size;
        if pp_size > 1 { pp_size } else { 1 }
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> EngineResult<()> {
        let responses = self
            .collective_rpc_blocking(|| WorkerRequest::InitializeCache {
                num_gpu_blocks,
                num_cpu_blocks,
            })
            .map_err(|e| vllm_engine::error::EngineError::Executor(e.to_string()))?;

        // Check all succeeded.
        for response in responses {
            match response {
                WorkerResponse::CacheInitialized(Ok(())) => {}
                WorkerResponse::CacheInitialized(Err(e)) => {
                    return Err(vllm_engine::error::EngineError::Executor(e.to_string()));
                }
                other => {
                    return Err(vllm_engine::error::EngineError::Executor(format!(
                        "unexpected response: {other:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn determine_available_memory(&mut self) -> EngineResult<Vec<usize>> {
        let responses = self
            .collective_rpc_blocking(|| WorkerRequest::DetermineAvailableMemory)
            .map_err(|e| vllm_engine::error::EngineError::Executor(e.to_string()))?;

        let mut memories = Vec::with_capacity(responses.len());
        for response in responses {
            match response {
                WorkerResponse::AvailableMemory(Ok(mem)) => memories.push(mem),
                WorkerResponse::AvailableMemory(Err(e)) => {
                    return Err(vllm_engine::error::EngineError::Executor(e.to_string()));
                }
                other => {
                    return Err(vllm_engine::error::EngineError::Executor(format!(
                        "unexpected response: {other:?}"
                    )));
                }
            }
        }
        Ok(memories)
    }

    fn check_health(&self) -> EngineResult<()> {
        let responses = self
            .collective_rpc_blocking(|| WorkerRequest::CheckHealth)
            .map_err(|e| vllm_engine::error::EngineError::Executor(e.to_string()))?;

        for response in responses {
            match response {
                WorkerResponse::HealthOk(Ok(())) => {}
                WorkerResponse::HealthOk(Err(e)) => {
                    return Err(vllm_engine::error::EngineError::Executor(e.to_string()));
                }
                other => {
                    return Err(vllm_engine::error::EngineError::Executor(format!(
                        "unexpected response: {other:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn sleep(&mut self, level: u32) -> EngineResult<()> {
        if self.is_sleeping {
            return Ok(());
        }
        let responses = self
            .collective_rpc_blocking(|| WorkerRequest::Sleep(level))
            .map_err(|e| vllm_engine::error::EngineError::Executor(e.to_string()))?;

        for response in responses {
            if let WorkerResponse::SleepDone(Err(e)) = response {
                return Err(vllm_engine::error::EngineError::Executor(e.to_string()));
            }
        }
        self.is_sleeping = true;
        Ok(())
    }

    fn wake_up(&mut self, tags: Option<&[String]>) -> EngineResult<()> {
        if !self.is_sleeping {
            return Ok(());
        }
        let tags_owned = tags.map(|t| t.to_vec());
        let responses = self
            .collective_rpc_blocking(|| WorkerRequest::WakeUp(tags_owned.clone()))
            .map_err(|e| vllm_engine::error::EngineError::Executor(e.to_string()))?;

        for response in responses {
            if let WorkerResponse::WakeUpDone(Err(e)) = response {
                return Err(vllm_engine::error::EngineError::Executor(e.to_string()));
            }
        }
        self.is_sleeping = false;
        Ok(())
    }

    fn is_sleeping(&self) -> bool {
        self.is_sleeping
    }

    fn shutdown(&mut self) {
        if self.is_shutdown {
            return;
        }
        info!(
            "MultiprocExecutor: shutting down {} workers",
            self.workers.len()
        );

        // Send shutdown to all workers (fire-and-forget).
        let _ = self.collective_rpc_blocking(|| WorkerRequest::Shutdown);
        self.is_shutdown = true;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::{NoopWorker, WorkerConfig};
    use std::collections::HashMap;

    fn make_workers(count: usize) -> Vec<Box<dyn Worker>> {
        (0..count)
            .map(|rank| {
                let config = WorkerConfig {
                    local_rank: rank,
                    rank,
                    is_driver_worker: rank == 0,
                    distributed_init_method: "tcp://localhost:0".to_string(),
                };
                let mut worker = NoopWorker::new(config, 16 * 1024 * 1024);
                worker.init_device().unwrap();
                worker.load_model().unwrap();
                Box::new(worker) as Box<dyn Worker>
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_multiproc_single_worker() {
        let workers = make_workers(1);
        let parallel_config = ResolvedParallelConfig::single_gpu();
        let runtime = tokio::runtime::Handle::current();

        let mut exec = MultiprocExecutor::new(workers, parallel_config, runtime);
        assert_eq!(exec.num_workers(), 1);
        assert!(!exec.is_failed());

        // Initialize cache.
        exec.initialize_cache(512, 0).unwrap();

        // Execute model.
        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("req-1".to_string(), 100);

        let sched_output = SchedulerOutput {
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 100,
            ..SchedulerOutput::make_empty()
        };

        let output = exec.execute_model(&sched_output).unwrap();
        assert_eq!(output.num_requests(), 1);
        assert!(output.get_tokens("req-1").is_some());

        exec.shutdown();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_multiproc_multiple_workers() {
        let workers = make_workers(4);
        let parallel_config = ResolvedParallelConfig::tensor_parallel(4, 0);
        let runtime = tokio::runtime::Handle::current();

        let mut exec = MultiprocExecutor::new(workers, parallel_config, runtime);
        assert_eq!(exec.num_workers(), 4);

        // Determine available memory from all workers.
        let memories = exec.determine_available_memory().unwrap();
        assert_eq!(memories.len(), 4);
        for mem in &memories {
            assert_eq!(*mem, 16 * 1024 * 1024);
        }

        // Health check.
        exec.check_health().unwrap();

        exec.shutdown();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_multiproc_tp_pp() {
        // TP=2, PP=2 → 4 workers
        let workers = make_workers(4);
        let parallel_config = ResolvedParallelConfig::tensor_pipeline_parallel(2, 2, 0);
        let runtime = tokio::runtime::Handle::current();

        let mut exec = MultiprocExecutor::new(workers, parallel_config, runtime);

        // PP size > 1 → max_concurrent_batches = pp_size = 2
        assert_eq!(exec.max_concurrent_batches(), 2);

        // Execute model — result comes from output rank (rank 2 = TP0_PP1).
        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("req-1".to_string(), 50);

        let sched_output = SchedulerOutput {
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 50,
            ..SchedulerOutput::make_empty()
        };

        let output = exec.execute_model(&sched_output).unwrap();
        assert_eq!(output.num_requests(), 1);

        exec.shutdown();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_multiproc_sleep_wake() {
        let workers = make_workers(2);
        let parallel_config = ResolvedParallelConfig::tensor_parallel(2, 0);
        let runtime = tokio::runtime::Handle::current();

        let mut exec = MultiprocExecutor::new(workers, parallel_config, runtime);

        assert!(!exec.is_sleeping());
        exec.sleep(1).unwrap();
        assert!(exec.is_sleeping());

        // Double sleep is a no-op.
        exec.sleep(1).unwrap();

        exec.wake_up(None).unwrap();
        assert!(!exec.is_sleeping());

        exec.shutdown();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_multiproc_shutdown_idempotent() {
        let workers = make_workers(2);
        let parallel_config = ResolvedParallelConfig::tensor_parallel(2, 0);
        let runtime = tokio::runtime::Handle::current();

        let mut exec = MultiprocExecutor::new(workers, parallel_config, runtime);
        exec.shutdown();
        exec.shutdown(); // Should not panic.
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_multiproc_initialize_cache_all() {
        let workers = make_workers(3);
        let parallel_config = ResolvedParallelConfig::tensor_parallel(3, 0);
        let runtime = tokio::runtime::Handle::current();

        let mut exec = MultiprocExecutor::new(workers, parallel_config, runtime);
        exec.initialize_cache(1024, 0).unwrap();

        exec.shutdown();
    }
}
