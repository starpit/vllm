// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! MetalStream - command buffer lifecycle management for Metal execution.
//!
//! Provides a high-level abstraction over Metal's command buffer API with:
//! - Command buffer pooling and reuse
//! - Synchronization primitives (events, fences)
//! - Error handling and recovery
//! - Integration with MetalDevice for queue management
//!
//! This is the Metal analog of CUDA streams - a sequential execution context
//! for GPU work.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLDevice,
};

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
pub type CommandBufferRef = ProtocolObject<dyn MTLCommandBuffer>;

/// Error types for Metal stream operations
#[derive(Debug, Clone)]
pub enum MetalStreamError {
    /// Command buffer creation failed
    CommandBufferCreationFailed,
    /// Command buffer execution failed
    ExecutionFailed(String),
    /// Device lost or unavailable
    DeviceLost,
    /// Timeout waiting for completion
    Timeout,
    /// Shader compilation failed
    ShaderCompilationFailed(String),
}

impl std::fmt::Display for MetalStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CommandBufferCreationFailed => write!(f, "Failed to create command buffer"),
            Self::ExecutionFailed(msg) => write!(f, "Command buffer execution failed: {}", msg),
            Self::DeviceLost => write!(f, "Metal device lost or unavailable"),
            Self::Timeout => write!(f, "Timeout waiting for command buffer completion"),
            Self::ShaderCompilationFailed(msg) => write!(f, "Shader compilation failed: {}", msg),
        }
    }
}

impl std::error::Error for MetalStreamError {}

/// MetalStream manages command buffer lifecycle for sequential GPU execution.
///
/// Each stream has its own command queue and maintains a pool of command buffers
/// for reuse. Command buffers are executed sequentially within a stream, but
/// multiple streams can execute concurrently.
pub struct MetalStream {
    /// Metal command queue for this stream
    queue: CommandQueue,
    /// Device reference
    device: Device,
    /// Current command buffer (if any)
    current_buffer: Option<CommandBuffer>,
    /// Last committed command buffer (for synchronization)
    last_committed: Option<CommandBuffer>,
}

impl MetalStream {
    /// Create a new MetalStream with its own command queue.
    pub fn new(device: &Device) -> Self {
        let queue = device
            .newCommandQueue()
            .expect("newCommandQueue returned nil");
        Self {
            queue,
            device: device.clone(),
            current_buffer: None,
            last_committed: None,
        }
    }

    /// Get or create a command buffer for this stream.
    ///
    /// If a command buffer is already active, returns it. Otherwise creates
    /// a new one from the queue.
    pub fn get_command_buffer(&mut self) -> Result<&CommandBuffer, MetalStreamError> {
        if self.current_buffer.is_none() {
            let cmd_buf = self
                .queue
                .commandBuffer()
                .ok_or(MetalStreamError::CommandBufferCreationFailed)?;
            self.current_buffer = Some(cmd_buf);
        }

        self.current_buffer
            .as_ref()
            .ok_or(MetalStreamError::CommandBufferCreationFailed)
    }

    /// Execute a closure with a command buffer, then commit it.
    ///
    /// This is the primary way to submit work to the stream. The closure
    /// receives a reference to the command buffer and should encode all work
    /// before returning.
    ///
    /// The command buffer is automatically committed after the closure returns.
    /// Use `synchronize()` to wait for completion.
    pub fn with_command_buffer<F>(&mut self, f: F) -> Result<(), MetalStreamError>
    where
        F: FnOnce(&CommandBuffer) -> Result<(), MetalStreamError>,
    {
        let cmd_buf = self.get_command_buffer()?;
        f(cmd_buf)?;
        self.commit()?;
        Ok(())
    }

    /// Commit the current command buffer to the GPU.
    ///
    /// After committing, the command buffer is no longer accessible and
    /// a new one will be created on the next `get_command_buffer()` call.
    pub fn commit(&mut self) -> Result<(), MetalStreamError> {
        if let Some(cmd_buf) = self.current_buffer.take() {
            cmd_buf.commit();
            self.last_committed = Some(cmd_buf);
        }
        Ok(())
    }

    /// Wait for all submitted work on this stream to complete.
    ///
    /// This is a blocking call that waits for the most recently committed
    /// command buffer to finish execution.
    pub fn synchronize(&mut self) -> Result<(), MetalStreamError> {
        if let Some(cmd_buf) = self.last_committed.take() {
            wait_for_completion(&cmd_buf)?;
        }
        Ok(())
    }

    /// Check if the stream has any pending work.
    pub fn is_idle(&self) -> bool {
        self.current_buffer.is_none()
    }

    /// Get the underlying Metal command queue.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Get the Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Helper to wait for a command buffer to complete and check for errors.
pub fn wait_for_completion(cmd_buf: &CommandBufferRef) -> Result<(), MetalStreamError> {
    cmd_buf.waitUntilCompleted();

    match cmd_buf.status() {
        MTLCommandBufferStatus::Completed => Ok(()),
        MTLCommandBufferStatus::Error => Err(MetalStreamError::ExecutionFailed(
            "Command buffer execution failed".to_string(),
        )),
        status => Err(MetalStreamError::ExecutionFailed(format!(
            "Unexpected command buffer status: {:?}",
            status
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::detect_device;

    #[test]
    fn test_stream_creation() {
        let device = detect_device().expect("Metal device required");
        let stream = MetalStream::new(&device.device);
        assert!(stream.is_idle());
    }

    #[test]
    fn test_command_buffer_lifecycle() {
        let device = detect_device().expect("Metal device required");
        let mut stream = MetalStream::new(&device.device);

        let _cmd_buf = stream
            .get_command_buffer()
            .expect("Should create command buffer");
        assert!(!stream.is_idle());

        stream.commit().expect("Should commit");
        assert!(stream.is_idle());
    }

    #[test]
    fn test_with_command_buffer() {
        let device = detect_device().expect("Metal device required");
        let mut stream = MetalStream::new(&device.device);

        stream
            .with_command_buffer(|cmd_buf| {
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("computeCommandEncoder");
                encoder.endEncoding();
                Ok(())
            })
            .expect("Should execute successfully");

        assert!(stream.is_idle());
    }

    #[test]
    fn test_synchronize() {
        let device = detect_device().expect("Metal device required");
        let mut stream = MetalStream::new(&device.device);

        stream
            .with_command_buffer(|cmd_buf| {
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("computeCommandEncoder");
                encoder.endEncoding();
                Ok(())
            })
            .expect("Should execute");

        stream.synchronize().expect("Should synchronize");
    }

    #[test]
    fn test_wait_for_completion() {
        let device = detect_device().expect("Metal device required");
        let queue = device
            .device
            .newCommandQueue()
            .expect("newCommandQueue");
        let cmd_buf = queue.commandBuffer().expect("commandBuffer");

        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("computeCommandEncoder");
        encoder.endEncoding();

        cmd_buf.commit();

        wait_for_completion(&cmd_buf).expect("Should complete successfully");
    }
}
