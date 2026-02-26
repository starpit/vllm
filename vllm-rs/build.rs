// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Workspace-level build script.
//!
//! This build script will be extended to handle CUDA kernel compilation
//! when the `vllm-kernels` crate is brought up. For now, it's a no-op
//! placeholder.

fn main() {
    // Future: CUDA kernel compilation via the `cc` crate.
    // See vllm-rs/crates/vllm-kernels/build.rs for per-crate builds.
    println!("cargo:rerun-if-changed=build.rs");
}
