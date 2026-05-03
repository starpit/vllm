// SPDX-License-Identifier: Apache-2.0
//! See `ferrite-model-mistral/build.rs` for the rationale.

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
