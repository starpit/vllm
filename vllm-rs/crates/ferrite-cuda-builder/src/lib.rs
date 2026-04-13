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
