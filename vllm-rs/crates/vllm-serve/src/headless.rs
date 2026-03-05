// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Headless worker loop for remote (non-master) nodes in multi-node TP.
//!
//! After NCCL init, node_rank > 0 enters this blocking loop instead of
//! starting an engine/scheduler/HTTP server. The master node sends
//! `ControlMessage`s over TCP; this loop deserializes them and dispatches
//! to the local `ThreadPoolExecutor`. Outputs are discarded — only the
//! master node returns results to clients. NCCL collectives synchronize
//! the forward passes across nodes.

use std::net::TcpStream;

use tracing::{error, info};
use vllm_engine::executor::Executor;
use vllm_executor::multinode::{ControlMessage, recv_message};
use vllm_executor::threadpool::ThreadPoolExecutor;

/// Run the headless worker loop. Blocks forever (or until shutdown).
///
/// Connects to the master's control port, then loops: receive scheduler
/// output → execute locally → discard output → repeat.
pub fn run_headless(
    mut executor: ThreadPoolExecutor,
    master_addr: &str,
    control_port: u16,
) -> anyhow::Result<()> {
    let addr = format!("{master_addr}:{control_port}");
    info!("Headless worker: connecting to master at {addr}");
    let mut stream = TcpStream::connect(&addr)?;
    stream.set_nodelay(true)?;
    info!("Headless worker: connected to master");

    loop {
        let msg = match recv_message(&mut stream) {
            Ok(msg) => msg,
            Err(e) => {
                // Connection closed or error — master shut down.
                info!("Headless worker: connection closed ({e}), shutting down");
                break;
            }
        };

        match msg {
            ControlMessage::ExecuteModel(scheduler_output) => {
                if let Err(e) = executor.execute_model(&scheduler_output) {
                    error!("Headless worker: execute_model failed: {e}");
                }
                // Output discarded — master uses its own rank's output.
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
