// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Multi-node executor: wraps a local `ThreadPoolExecutor` and broadcasts
//! `SchedulerOutput` to remote headless worker nodes over TCP before each
//! forward step. Remote nodes execute the same forward pass so NCCL
//! collectives (all-reduce, all-gather) synchronize correctly across nodes.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use serde::{Deserialize, Serialize};
use tracing::{error, info};
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_engine::error::{EngineError, EngineResult};
use vllm_engine::executor::{Executor, ModelRunnerOutput};

use crate::threadpool::ThreadPoolExecutor;

// ---------------------------------------------------------------------------
// Control protocol
// ---------------------------------------------------------------------------

/// Messages sent from the master node to remote headless workers.
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

/// Send a length-prefixed bincode message over a TCP stream.
pub fn send_message(stream: &mut TcpStream, msg: &ControlMessage) -> std::io::Result<()> {
    let data = bincode::serialize(msg)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let len = (data.len() as u64).to_le_bytes();
    stream.write_all(&len)?;
    stream.write_all(&data)?;
    stream.flush()
}

/// Receive a length-prefixed bincode message from a TCP stream.
pub fn recv_message(stream: &mut TcpStream) -> std::io::Result<ControlMessage> {
    let mut len_buf = [0u8; 8];
    stream.read_exact(&mut len_buf)?;
    let len = u64::from_le_bytes(len_buf) as usize;
    let mut data = vec![0u8; len];
    stream.read_exact(&mut data)?;
    bincode::deserialize(&data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
}

// ---------------------------------------------------------------------------
// MultiNodeExecutor
// ---------------------------------------------------------------------------

/// Wraps a local `ThreadPoolExecutor` and broadcasts scheduler outputs to
/// remote nodes before each local `execute_model` call.
pub struct MultiNodeExecutor {
    inner: ThreadPoolExecutor,
    remotes: Vec<TcpStream>,
}

impl MultiNodeExecutor {
    /// Wait for `num_remotes` headless workers to connect on `control_port`,
    /// then wrap the local executor.
    pub fn accept_remotes(
        inner: ThreadPoolExecutor,
        control_port: u16,
        num_remotes: usize,
    ) -> anyhow::Result<Self> {
        info!(
            "MultiNodeExecutor: waiting for {} remote node(s) on port {}",
            num_remotes, control_port
        );
        let listener = TcpListener::bind(("0.0.0.0", control_port))?;
        let mut remotes = Vec::with_capacity(num_remotes);
        for i in 0..num_remotes {
            let (stream, addr) = listener.accept()?;
            stream.set_nodelay(true)?;
            info!("Remote node {} connected from {}", i + 1, addr);
            remotes.push(stream);
        }
        info!("All {} remote nodes connected", num_remotes);
        Ok(Self { inner, remotes })
    }

    /// Broadcast a control message to all remote nodes.
    fn broadcast_to_remotes(&mut self, msg: &ControlMessage) -> EngineResult<()> {
        for (i, stream) in self.remotes.iter_mut().enumerate() {
            send_message(stream, msg).map_err(|e| {
                EngineError::Executor(format!("failed to send to remote node {i}: {e}"))
            })?;
        }
        Ok(())
    }
}

impl Executor for MultiNodeExecutor {
    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> EngineResult<ModelRunnerOutput> {
        // Broadcast to remote nodes first so they start their forward pass.
        let msg = ControlMessage::ExecuteModel(Box::new(scheduler_output.clone()));
        self.broadcast_to_remotes(&msg)?;
        // Then execute locally — NCCL collectives will synchronize.
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
        self.broadcast_to_remotes(&msg)?;
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
        // Tell remote nodes to shut down.
        let msg = ControlMessage::Shutdown;
        for (i, stream) in self.remotes.iter_mut().enumerate() {
            if let Err(e) = send_message(stream, &msg) {
                error!("Failed to send shutdown to remote node {i}: {e}");
            }
        }
        self.inner.shutdown();
    }
}
