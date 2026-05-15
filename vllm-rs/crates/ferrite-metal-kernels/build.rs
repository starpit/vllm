// SPDX-License-Identifier: Apache-2.0
//! Ahead-of-time Metal shader compilation.
//!
//! Every `.metal` file under `shaders/` is compiled to a per-library
//! `.metallib` in `OUT_DIR` so the runtime can `new_library_with_data`
//! a precompiled blob instead of paying the MSL→AIR frontend cost on
//! every process start. Shaders are independent (each maps to one
//! `library_name` key in `SpecializedPipelineCache`), so we keep one
//! `.metallib` per source file rather than one bundle.
//!
//! Function-constant specialization still happens at runtime through
//! `library.get_function(name, Some(constants))`; the AoT step replaces
//! only the source-compile + link stages (steps 1-2 of the Metal
//! pipeline). Step 4 (AIR → GPU machine code) remains lazy per
//! pipeline state.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Non-macOS targets skip the toolchain entirely — `ferrite-metal-kernels`
    // itself is `cfg(target_os = "macos")` at the call sites that load
    // the metallibs, so emitting nothing here lets Linux/cuda builds
    // compile this crate without needing `xcrun`.
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "macos" {
        return;
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest_dir.join("shaders");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=build.rs");

    let mut entries: Vec<_> = std::fs::read_dir(&shader_dir)
        .unwrap_or_else(|e| panic!("read shaders/ failed: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("metal"))
        .collect();
    entries.sort();

    for shader in &entries {
        let stem = shader.file_stem().unwrap().to_str().unwrap();
        let air = out_dir.join(format!("{stem}.air"));
        let metallib = out_dir.join(format!("{stem}.metallib"));

        // MSL → AIR. `-O3` and `-frecord-sources=flat` so debug
        // captures retain source mapping; matches what MLX ships.
        let status = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-O3",
                "-frecord-sources=flat",
                "-c",
            ])
            .arg(shader)
            .arg("-o")
            .arg(&air)
            .status()
            .unwrap_or_else(|e| panic!("spawn `xcrun metal` failed: {e}"));
        if !status.success() {
            panic!("`xcrun metal` failed for {}", shader.display());
        }

        // AIR → metallib.
        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib"])
            .arg(&air)
            .arg("-o")
            .arg(&metallib)
            .status()
            .unwrap_or_else(|e| panic!("spawn `xcrun metallib` failed: {e}"));
        if !status.success() {
            panic!("`xcrun metallib` failed for {}", shader.display());
        }
    }
}
