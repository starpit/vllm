// SPDX-License-Identifier: Apache-2.0
//! See `ferrite-model-mistral/build.rs` for the rationale. Without
//! this, cargo never learns the proc-macro reads `FERRITE_MODELS` at
//! expansion time, so a filter change (or set->unset) does NOT
//! recompile this crate — it silently reuses the last rlib. For a
//! VL/MM crate that means a stem filtered out by an earlier build
//! stays compiled to an empty crate (no `VisionArchWeights` impl, no
//! inventory row), and the model loads TEXT-ONLY until the crate is
//! hand-touched. (These 5 VL/MM crates lacked a build.rs because they
//! were cuda-gated; metal builds exposed the gap.)

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
