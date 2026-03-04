// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Thread-pool executor for multi-GPU tensor parallelism.
//!
//! Unlike `MultiprocExecutor` (which uses tokio tasks), this executor gives
//! each worker a dedicated OS thread. This is essential for NCCL: collectives
//! like all-reduce block until all ranks participate, so each rank must run
//! on its own OS thread to avoid deadlock.
//!
//! # Architecture
//!
//! ```text
//!   ThreadPoolExecutor
//!     ├── OS Thread (rank 0) ──► CandleWorker + NcclProcessGroup
//!     └── OS Thread (rank 1) ──► CandleWorker + NcclProcessGroup
//! ```
//!
//! The executor broadcasts scheduler outputs to all workers via channels,
//! and workers signal completion via response channels. Only the output
//! rank's result is returned to the engine.

use std::sync::Arc;

use tracing::info;
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_engine::error::EngineResult;
use vllm_engine::executor::{Executor, ModelRunnerOutput};

use crate::error::ExecutorResult;
use crate::parallel::ResolvedParallelConfig;
use crate::worker::Worker;

// ---------------------------------------------------------------------------
// Worker handle (OS thread + channels)
// ---------------------------------------------------------------------------

/// Message sent from executor to worker thread.
enum Request {
    ExecuteModel(Arc<SchedulerOutput>),
    InitializeCache {
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    },
    DetermineAvailableMemory,
    CheckHealth,
    Shutdown,
}

/// Response from worker thread to executor.
enum Response {
    ModelOutput(Box<ExecutorResult<ModelRunnerOutput>>),
    CacheInitialized(ExecutorResult<()>),
    AvailableMemory(ExecutorResult<usize>),
    HealthOk(ExecutorResult<()>),
    ShutdownAck,
}

struct WorkerThread {
    #[allow(dead_code)]
    rank: usize,
    is_output_rank: bool,
    tx: std::sync::mpsc::Sender<Request>,
    rx: std::sync::mpsc::Receiver<Response>,
    _handle: std::thread::JoinHandle<()>,
}

fn worker_loop(
    mut worker: Box<dyn Worker>,
    rx: std::sync::mpsc::Receiver<Request>,
    tx: std::sync::mpsc::Sender<Response>,
) {
    while let Ok(request) = rx.recv() {
        let response = match request {
            Request::ExecuteModel(sched_output) => {
                Response::ModelOutput(Box::new(worker.execute_model(&sched_output)))
            }
            Request::InitializeCache {
                num_gpu_blocks,
                num_cpu_blocks,
            } => {
                Response::CacheInitialized(worker.initialize_cache(num_gpu_blocks, num_cpu_blocks))
            }
            Request::DetermineAvailableMemory => {
                Response::AvailableMemory(worker.determine_available_memory())
            }
            Request::CheckHealth => Response::HealthOk(worker.check_health()),
            Request::Shutdown => {
                worker.shutdown();
                let _ = tx.send(Response::ShutdownAck);
                return;
            }
        };
        if tx.send(response).is_err() {
            return; // Executor dropped
        }
    }
}

// ---------------------------------------------------------------------------
// ThreadPoolExecutor
// ---------------------------------------------------------------------------

/// Multi-GPU executor using dedicated OS threads per worker.
///
/// Each worker runs on its own OS thread, ensuring NCCL collectives can
/// execute concurrently across ranks without tokio scheduling issues.
pub struct ThreadPoolExecutor {
    workers: Vec<WorkerThread>,
    #[allow(dead_code)]
    parallel_config: ResolvedParallelConfig,
    is_shutdown: bool,
}

impl ThreadPoolExecutor {
    /// Create a new thread-pool executor from pre-initialized workers.
    ///
    /// Workers are moved to dedicated OS threads. They must already have
    /// device init, model load, and NCCL injection completed.
    pub fn new(workers: Vec<Box<dyn Worker>>, parallel_config: ResolvedParallelConfig) -> Self {
        let output_rank = parallel_config.output_rank();

        let handles: Vec<WorkerThread> = workers
            .into_iter()
            .enumerate()
            .map(|(rank, worker)| {
                let (req_tx, req_rx) = std::sync::mpsc::channel();
                let (resp_tx, resp_rx) = std::sync::mpsc::channel();

                let handle = std::thread::Builder::new()
                    .name(format!("vllm-worker-{rank}"))
                    .spawn(move || worker_loop(worker, req_rx, resp_tx))
                    .expect("failed to spawn worker thread");

                WorkerThread {
                    rank,
                    is_output_rank: rank == output_rank,
                    tx: req_tx,
                    rx: resp_rx,
                    _handle: handle,
                }
            })
            .collect();

        info!(
            "ThreadPoolExecutor: spawned {} worker threads, output_rank={}",
            handles.len(),
            output_rank
        );

        Self {
            workers: handles,
            parallel_config,
            is_shutdown: false,
        }
    }

    /// Broadcast a request to all workers and collect responses.
    fn broadcast(&self, make_request: impl Fn() -> Request) -> Vec<Response> {
        // Send to all workers.
        for w in &self.workers {
            let _ = w.tx.send(make_request());
        }
        // Collect responses (blocking — each worker responds in order).
        self.workers.iter().map(|w| w.rx.recv().unwrap()).collect()
    }
}

impl Executor for ThreadPoolExecutor {
    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> EngineResult<ModelRunnerOutput> {
        let shared = Arc::new(scheduler_output.clone());
        let responses = self.broadcast(|| Request::ExecuteModel(Arc::clone(&shared)));

        // Return output rank's result.
        for (w, resp) in self.workers.iter().zip(responses) {
            if w.is_output_rank {
                return match resp {
                    Response::ModelOutput(result) => (*result)
                        .map_err(|e| vllm_engine::error::EngineError::Executor(e.to_string())),
                    _ => Err(vllm_engine::error::EngineError::Executor(
                        "unexpected response".to_string(),
                    )),
                };
            }
        }
        Err(vllm_engine::error::EngineError::Executor(
            "no output rank".to_string(),
        ))
    }

    fn max_concurrent_batches(&self) -> usize {
        1
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> EngineResult<()> {
        let responses = self.broadcast(|| Request::InitializeCache {
            num_gpu_blocks,
            num_cpu_blocks,
        });
        for resp in responses {
            if let Response::CacheInitialized(Err(e)) = resp {
                return Err(vllm_engine::error::EngineError::Executor(e.to_string()));
            }
        }
        Ok(())
    }

    fn determine_available_memory(&mut self) -> EngineResult<Vec<usize>> {
        let responses = self.broadcast(|| Request::DetermineAvailableMemory);
        let mut memories = Vec::with_capacity(responses.len());
        for resp in responses {
            match resp {
                Response::AvailableMemory(Ok(mem)) => memories.push(mem),
                Response::AvailableMemory(Err(e)) => {
                    return Err(vllm_engine::error::EngineError::Executor(e.to_string()));
                }
                _ => {
                    return Err(vllm_engine::error::EngineError::Executor(
                        "unexpected response".to_string(),
                    ));
                }
            }
        }
        Ok(memories)
    }

    fn check_health(&self) -> EngineResult<()> {
        // Send to all workers.
        for w in &self.workers {
            let _ = w.tx.send(Request::CheckHealth);
        }
        for w in &self.workers {
            match w.rx.recv().unwrap() {
                Response::HealthOk(Ok(())) => {}
                Response::HealthOk(Err(e)) => {
                    return Err(vllm_engine::error::EngineError::Executor(e.to_string()));
                }
                _ => {
                    return Err(vllm_engine::error::EngineError::Executor(
                        "unexpected response".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn sleep(&mut self, _level: u32) -> EngineResult<()> {
        Ok(()) // Not implemented for thread pool
    }

    fn wake_up(&mut self, _tags: Option<&[String]>) -> EngineResult<()> {
        Ok(()) // Not implemented for thread pool
    }

    fn is_sleeping(&self) -> bool {
        false
    }

    fn shutdown(&mut self) {
        if self.is_shutdown {
            return;
        }
        info!(
            "ThreadPoolExecutor: shutting down {} workers",
            self.workers.len()
        );
        for w in &self.workers {
            let _ = w.tx.send(Request::Shutdown);
        }
        // Wait for acks.
        for w in &self.workers {
            let _ = w.rx.recv();
        }
        self.is_shutdown = true;
    }
}
