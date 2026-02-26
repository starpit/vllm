// SPDX-License-Identifier: Apache-2.0
//! FFI bindings to CUDA/C++ kernels and CPU fallback implementations.
//!
//! This crate provides trait-based abstractions for GPU kernels used in
//! vLLM inference. Each trait defines the kernel interface, with:
//! - **CPU implementations** for testing without GPU hardware
//! - **CUDA FFI bindings** (behind `cuda` feature) for production use
//!
//! Port of: kernel functions declared in `csrc/ops.h` and `csrc/cache.h`

pub mod activation;
pub mod attention;
pub mod cache;
pub mod error;
pub mod norm;
pub mod rotary;

pub use error::{KernelError, KernelResult};
