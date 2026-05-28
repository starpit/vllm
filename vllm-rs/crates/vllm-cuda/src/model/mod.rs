// SPDX-License-Identifier: Apache-2.0
//! Hand-written CUDA model forwards have been removed; ferrite-forward is
//! the sole model-forward path. This module survives only as a re-export
//! shim for `attention_helpers`, which the `forward!()` codegen still
//! resolves through `vllm_cuda::model::attention_helpers`.

pub use ferrite_kernels::attention_helpers;
