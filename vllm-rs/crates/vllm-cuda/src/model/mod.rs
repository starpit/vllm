// SPDX-License-Identifier: Apache-2.0
//! Model implementations using `GpuTensor`.

// Re-export from ferrite-kernels so `crate::model::attention_helpers` still resolves
// (needed by forward!() codegen until Phase 3 updates the paths).
pub use ferrite_kernels::attention_helpers;
pub mod commandr;
pub mod deepseek_v2;
pub mod gemma2;
pub mod gemma3;
pub mod llama;
pub mod mixtral;
pub mod qwen2;
pub mod qwen2_moe;
pub mod qwen3_moe;
pub mod qwen3_next;
