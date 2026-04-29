// SPDX-License-Identifier: Apache-2.0
//! Tells cargo to invalidate this crate's compilation cache when
//! the env vars the proc-macro reads at expansion time change.

fn main() {
    println!("cargo:rerun-if-env-changed=FERRITE_MODELS");
    println!("cargo:rerun-if-env-changed=FERRITE_GPU");
}
