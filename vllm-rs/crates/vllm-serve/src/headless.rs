// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Headless worker loop for remote (non-master) nodes in multi-node TP.
//!
//! After NCCL init, node_rank > 0 enters this blocking loop instead of
//! starting an engine/scheduler/HTTP server. Rank 0 broadcasts
//! `ControlMessage`s via NCCL; this loop receives them (on the worker
//! thread, which owns the correct CUDA context) and dispatches to the
//! local `ThreadPoolExecutor`.

use tracing::{error, info};
use vllm_engine::executor::Executor;
use vllm_executor::multinode::ControlMessage;
use vllm_executor::threadpool::ThreadPoolExecutor;

/// Run the headless worker loop. Blocks forever (or until shutdown).
///
/// Loops on NCCL broadcast (via worker thread): receive ControlMessage
/// from rank 0, dispatch to local executor, discard output, repeat.
pub fn run_headless(mut executor: ThreadPoolExecutor) -> anyhow::Result<()> {
    info!("Headless worker: entering NCCL broadcast loop");

    loop {
        // Receive broadcast from rank 0 via NCCL (on the worker thread).
        let data = executor
            .nccl_broadcast(&[], 0)
            .map_err(|e| anyhow::anyhow!("NCCL broadcast recv failed: {e}"))?;

        let msg: ControlMessage = bincode::deserialize(&data)
            .map_err(|e| anyhow::anyhow!("failed to deserialize control message: {e}"))?;

        match msg {
            ControlMessage::ExecuteModel(scheduler_output) => {
                if let Err(e) = executor.execute_model(&scheduler_output) {
                    error!("Headless worker: execute_model failed: {e}");
                }
            }
            ControlMessage::InitCache {
                num_gpu_blocks,
                num_cpu_blocks,
            } => {
                info!(
                    "Headless worker: initializing cache (gpu_blocks={}, cpu_blocks={})",
                    num_gpu_blocks, num_cpu_blocks
                );
                if let Err(e) = executor.initialize_cache(num_gpu_blocks, num_cpu_blocks) {
                    error!("Headless worker: initialize_cache failed: {e}");
                }
            }
            ControlMessage::Shutdown => {
                info!("Headless worker: received shutdown, exiting");
                break;
            }
        }
    }

    executor.shutdown();
    Ok(())
}
