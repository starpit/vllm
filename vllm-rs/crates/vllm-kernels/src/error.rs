// SPDX-License-Identifier: Apache-2.0
//! Kernel error types.

use thiserror::Error;

pub type KernelResult<T> = Result<T, KernelError>;

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("shape error: {0}")]
    Shape(String),

    #[error("dtype error: {0}")]
    DType(String),

    #[error("CUDA not available")]
    CudaNotAvailable,

    #[error("kernel error: {0}")]
    Other(String),
}
