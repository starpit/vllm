// SPDX-License-Identifier: Apache-2.0
//! Tells cargo to invalidate this crate's compilation cache when
//! the env vars the proc-macro reads at expansion time change:
//!
//! - `FERRITE_MODELS` — restricts which model configs get compiled
//!   (see `vllm-rs/CLAUDE.md`).
//! - `FERRITE_GPU` — selects the target GPU profile from
//!   `ferrite-cuda-targets` (overrides nvidia-smi auto-detect).
//! - `FERRITE_DISABLE_CUBLAS_GEMM` — A/B hook that drops `GemmRefImpl`
//!   from the impl library so every singleton Gemm tile lands on
//!   CUTLASS. Used to drive the cuBLAS-freedom workstream.
//!
//! Env vars the proc-macro reads at expansion time aren't visible to
//! cargo's compilation-cache hashing on their own — `rerun-if-env-
//! changed` only re-runs the build script. To force an actual
//! recompile when the env var changes, we re-emit each value as
//! `cargo:rustc-env=KEY=VALUE`, which cargo includes in the rustc
//! invocation hash (and `sccache` propagates).

fn main() {
    forward_env("FERRITE_MODELS");
    forward_env("FERRITE_GPU");
    forward_env("FERRITE_DISABLE_CUBLAS_GEMM");
}

fn forward_env(name: &str) {
    println!("cargo:rerun-if-env-changed={name}");
    if let Ok(v) = std::env::var(name) {
        println!("cargo:rustc-env={name}={v}");
    }
}
