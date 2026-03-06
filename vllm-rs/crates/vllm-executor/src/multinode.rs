// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Multi-node executor: wraps a local `ThreadPoolExecutor` and broadcasts
//! `SchedulerOutput` to all ranks via NCCL before each forward step.
//!
//! Uses NCCL broadcast (a collective) routed through the worker thread
//! (which owns the correct CUDA context). Eliminates TCP control channel.

use serde::{Deserialize, Serialize};
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_engine::error::{EngineError, EngineResult};
use vllm_engine::executor::{Executor, ModelRunnerOutput};

use crate::threadpool::ThreadPoolExecutor;

// ---------------------------------------------------------------------------
// Control protocol (serialized via bincode, broadcast via NCCL)
// ---------------------------------------------------------------------------

/// Messages broadcast from rank 0 to all ranks via NCCL.
#[derive(Serialize, Deserialize)]
pub enum ControlMessage {
    /// Execute a forward pass with this scheduler output.
    ExecuteModel(Box<SchedulerOutput>),
    /// Initialize KV cache on all workers.
    InitCache {
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    },
    /// Shut down the remote worker loop.
    Shutdown,
}

// ---------------------------------------------------------------------------
// MultiNodeExecutor
// ---------------------------------------------------------------------------

/// Wraps a local `ThreadPoolExecutor` and broadcasts scheduler outputs to
/// all ranks via NCCL before each local `execute_model` call.
///
/// NCCL broadcast is dispatched to the worker thread (correct CUDA context)
/// via `ThreadPoolExecutor::nccl_broadcast`.
pub struct MultiNodeExecutor {
    inner: ThreadPoolExecutor,
}

impl MultiNodeExecutor {
    pub fn new(inner: ThreadPoolExecutor) -> Self {
        Self { inner }
    }

    /// Serialize and broadcast a control message via NCCL (rank 0 sends).
    fn broadcast_msg(&self, msg: &ControlMessage) -> EngineResult<()> {
        let data = bincode::serialize(msg).map_err(|e| {
            EngineError::Executor(format!("failed to serialize control message: {e}"))
        })?;
        self.inner
            .nccl_broadcast(&data, 0)
            .map_err(|e| EngineError::Executor(format!("NCCL broadcast failed: {e}")))?;
        Ok(())
    }
}

impl Executor for MultiNodeExecutor {
    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> EngineResult<ModelRunnerOutput> {
        // Broadcast to all ranks so remote workers start their forward pass.
        let msg = ControlMessage::ExecuteModel(Box::new(scheduler_output.clone()));
        self.broadcast_msg(&msg)?;
        // Execute locally — NCCL collectives in the model synchronize ranks.
        self.inner.execute_model(scheduler_output)
    }

    fn max_concurrent_batches(&self) -> usize {
        self.inner.max_concurrent_batches()
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> EngineResult<()> {
        let msg = ControlMessage::InitCache {
            num_gpu_blocks,
            num_cpu_blocks,
        };
        self.broadcast_msg(&msg)?;
        self.inner.initialize_cache(num_gpu_blocks, num_cpu_blocks)
    }

    fn determine_available_memory(&mut self) -> EngineResult<Vec<usize>> {
        self.inner.determine_available_memory()
    }

    fn check_health(&self) -> EngineResult<()> {
        self.inner.check_health()
    }

    fn sleep(&mut self, level: u32) -> EngineResult<()> {
        self.inner.sleep(level)
    }

    fn wake_up(&mut self, tags: Option<&[String]>) -> EngineResult<()> {
        self.inner.wake_up(tags)
    }

    fn is_sleeping(&self) -> bool {
        self.inner.is_sleeping()
    }

    fn shutdown(&mut self) {
        let _ = self.broadcast_msg(&ControlMessage::Shutdown);
        self.inner.shutdown();
    }
}
