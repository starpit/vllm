// SPDX-License-Identifier: Apache-2.0
//! AOT-compile MSL source to a `.metallib` blob via `xcrun metal -c`
//! and `xcrun metallib`. Shared by `ferrite-forward-macro` (proc-macro
//! expansion baking) and `ferrite-metal-cost-sweep` (benchmark of
//! the synthesized kernels).
//!
//! Same flow as `ferrite-metal-kernels/build.rs`.
//!
//! Panics on `xcrun` failure — synth compile errors are build errors
//! that need to surface, not runtime soft-fail.

use std::io::Write;
use std::process::Command;

pub fn aot_compile_metallib(symbol: &str, source: &str) -> Vec<u8> {
    // Skip on non-macOS hosts (no `xcrun`). The synth metallibs are
    // only ever consumed by the metal backend.
    let host_os =
        std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| std::env::consts::OS.to_string());
    if host_os != "macos" {
        return Vec::new();
    }

    let tmp_dir =
        std::env::temp_dir().join(format!("ferrite-synth-{}-{}", symbol, std::process::id()));
    std::fs::create_dir_all(&tmp_dir)
        .unwrap_or_else(|e| panic!("synth: create tmp dir {tmp_dir:?}: {e}"));

    let metal_path = tmp_dir.join(format!("{symbol}.metal"));
    let air_path = tmp_dir.join(format!("{symbol}.air"));
    let metallib_path = tmp_dir.join(format!("{symbol}.metallib"));

    let mut f = std::fs::File::create(&metal_path)
        .unwrap_or_else(|e| panic!("synth: create {metal_path:?}: {e}"));
    f.write_all(source.as_bytes())
        .unwrap_or_else(|e| panic!("synth: write {metal_path:?}: {e}"));
    drop(f);

    let status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-O3",
            "-frecord-sources=flat",
            "-c",
        ])
        .arg(&metal_path)
        .arg("-o")
        .arg(&air_path)
        .status()
        .unwrap_or_else(|e| panic!("synth: spawn xcrun metal: {e}"));
    if !status.success() {
        panic!("synth: `xcrun metal` failed for `{symbol}` (source at {metal_path:?})");
    }

    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg(&air_path)
        .arg("-o")
        .arg(&metallib_path)
        .status()
        .unwrap_or_else(|e| panic!("synth: spawn xcrun metallib: {e}"));
    if !status.success() {
        panic!("synth: `xcrun metallib` failed for `{symbol}`");
    }

    let bytes = std::fs::read(&metallib_path)
        .unwrap_or_else(|e| panic!("synth: read {metallib_path:?}: {e}"));

    if std::env::var("FERRITE_SYNTH_DUMP").is_ok() {
        let dump_dir = std::path::PathBuf::from("/tmp/ferrite-synth-dump");
        let _ = std::fs::create_dir_all(&dump_dir);
        let _ = std::fs::copy(&metal_path, dump_dir.join(format!("{symbol}.metal")));
        let _ = std::fs::copy(&metallib_path, dump_dir.join(format!("{symbol}.metallib")));
    }
    let _ = std::fs::remove_dir_all(&tmp_dir);
    bytes
}
