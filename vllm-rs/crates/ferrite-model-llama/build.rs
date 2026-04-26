// SPDX-License-Identifier: Apache-2.0
//! Tells cargo to invalidate this crate's compilation cache when
//! the env vars the proc-macro reads at expansion time change:
//!
//! - `FERRITE_MODELS` — restricts which model configs get compiled
//!   (see `vllm-rs/CLAUDE.md`).
//! - `FERRITE_GPU` — selects the target GPU profile from
//!   `ferrite-cuda-targets` (overrides nvidia-smi auto-detect).

fn main() {
    println!("cargo:rerun-if-env-changed=FERRITE_MODELS");
    println!("cargo:rerun-if-env-changed=FERRITE_GPU");
}
