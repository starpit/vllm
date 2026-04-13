// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Ferrite model architectures.
//!
//! Each model file contains a `forward!()` invocation that generates:
//! - `Model` struct with per-layer weights
//! - `Model::load()` for safetensors weight loading
//! - `Model::forward()` for the full forward pass
//! - Solver-generated dispatch functions per workload bucket

#[cfg(feature = "cuda")]
pub mod llama;
#[cfg(feature = "cuda")]
pub mod qwen2;
