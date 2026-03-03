// SPDX-License-Identifier: Apache-2.0
//! Process group trait for distributed collective communication.
//!
//! Defines the `ProcessGroup` trait used by tensor-parallel linear layers
//! to perform all-reduce and all-gather across GPU ranks.
//!
//! The concrete implementation (`NcclProcessGroup`) lives in `vllm-kernels`
//! to avoid circular dependencies. Layers reference it via `Arc<dyn ProcessGroup>`.

use candle_core::Tensor;

/// Abstraction over collective communication for tensor parallelism.
///
/// Implemented by `NcclProcessGroup` in `vllm-kernels` for NVIDIA GPUs.
/// Layers hold an `Option<Arc<dyn ProcessGroup>>` — when `None` (TP=1),
/// no communication is performed.
pub trait ProcessGroup: Send + Sync + std::fmt::Debug {
    /// Sum-reduce a tensor across all ranks (each rank gets the full result).
    fn all_reduce(&self, tensor: &Tensor) -> candle_core::Result<Tensor>;

    /// Gather a tensor from all ranks along `dim` (each rank gets the full result).
    fn all_gather(&self, tensor: &Tensor, dim: usize) -> candle_core::Result<Tensor>;

    /// This rank's index (0-based).
    fn rank(&self) -> usize;

    /// Total number of ranks.
    fn world_size(&self) -> usize;
}
