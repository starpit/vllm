// SPDX-License-Identifier: Apache-2.0
//! Tells cargo to invalidate this crate's compilation cache when
//! `FERRITE_MODELS` changes. The proc-macro reads that env var to
//! filter which model configs it processes.

fn main() {
    println!("cargo:rerun-if-env-changed=FERRITE_MODELS");
}
