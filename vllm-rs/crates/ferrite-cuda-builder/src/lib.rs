// SPDX-License-Identifier: Apache-2.0
//! CUDA kernel compilation for the ferrite inference framework.
//!
//! This crate has no runtime code. It exists solely for its `build.rs`,
//! which compiles all `.cu` files (static kernels + megakernels) into
//! static `.a` libraries. The dependency on `ferrite-models` ensures
//! that `forward!()` macro expansion runs first, generating the
//! megakernel `.cu` files before this build.rs tries to compile them.

// Re-export ferrite-models so downstream crates get the build ordering.
pub use ferrite_models;

// Per-tuple FlashInfer config set — symbol-name source of truth shared
// with `build.rs` (via `#[path]`) and consumed downstream by
// ferrite-kernels / ferrite-forward-macro when emitting FI call sites.
pub mod flashinfer_config;
