// SPDX-License-Identifier: Apache-2.0
//! Ahead-of-time Metal shader compilation + kernel-instantiation codegen.
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
//!
//! Codegen: a few kernels (currently `attention_steel_paged`) are
//! template-instantiated for one combo per (dtype, geometry-knob).
//! The instantiation list is the single source of truth in this file
//! and is mirrored to two generated files in `OUT_DIR`:
//!
//!   - `attention_steel_paged_instantiations.h` — included by the
//!     matching `.metal` source so `xcrun metal -I OUT_DIR` picks it
//!     up at MSL-compile time.
//!   - `steel_paged_kernels_generated.rs` — `include!`-ed by
//!     `kernel_identity.rs` (via the re-export below) so the
//!     dispatcher's symbol-lookup table comes from the same list.
//!
//! Adding a head-dim is a one-line edit to `STEEL_PAGED_HEAD_DIMS`
//! below.

use std::path::PathBuf;
use std::process::Command;

/// HEAD_DIMs (BD template arg of `attention_paged<...>` in
/// `mlx_steel_attn/steel_attention_paged_kernel.h`) instantiated in
/// `attention_steel_paged.metal`. Must cover every `head_dim`
/// reachable through the runtime gate in
/// `ferrite-forward/.../lowering.rs::AttentionPrefillPaged`. Sorted
/// by typical model frequency so the generated symbol table stays
/// readable.
const STEEL_PAGED_HEAD_DIMS: &[u32] = &[64, 96, 128, 256];

/// Activation dtypes the steel kernel is instantiated for. Tag is the
/// Rust/symbol-side spelling; type is the MSL spelling used in the
/// `INST_STEEL_PAGED` macro expansion.
const STEEL_PAGED_DTYPES: &[(&str, &str)] = &[
    ("f16",  "half"),
    ("bf16", "bfloat"),
];

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=build.rs");

    // Codegen runs on every host so the Rust side compiles on Linux/CUDA
    // pods even though no .metallib is produced there.
    write_steel_paged_instantiations_h(&out_dir);
    write_steel_paged_kernels_rs(&out_dir);

    // Non-macOS targets skip the toolchain entirely — `ferrite-metal-kernels`
    // itself is `cfg(target_os = "macos")` at the call sites that load
    // the metallibs, so emitting nothing here lets Linux/cuda builds
    // compile this crate without needing `xcrun`.
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "macos" {
        return;
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest_dir.join("shaders");

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
        // `-I OUT_DIR` so codegen-emitted headers (e.g.
        // `attention_steel_paged_instantiations.h`) resolve.
        let status = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-O3",
                "-frecord-sources=flat",
                "-I",
            ])
            .arg(&out_dir)
            .arg("-c")
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

/// Emit the `INST_STEEL_PAGED(tag, type, bd)` lines that
/// `attention_steel_paged.metal` includes. The `INST_STEEL_PAGED`
/// macro is defined in the .metal file itself; this header carries
/// only the per-combo expansions.
fn write_steel_paged_instantiations_h(out_dir: &std::path::Path) {
    let mut s = String::from(
        "// SPDX-License-Identifier: Apache-2.0\n\
         // Auto-generated by `ferrite-metal-kernels/build.rs`.\n\
         // Edit `STEEL_PAGED_HEAD_DIMS` in build.rs, not here.\n\n",
    );
    for &bd in STEEL_PAGED_HEAD_DIMS {
        for &(tag, ty) in STEEL_PAGED_DTYPES {
            s.push_str(&format!("INST_STEEL_PAGED({tag}, {ty}, {bd})\n"));
        }
    }
    std::fs::write(
        out_dir.join("attention_steel_paged_instantiations.h"),
        s,
    )
    .expect("write attention_steel_paged_instantiations.h");
}

/// Emit the Rust-side mirror: a slice of head-dims (for the runtime
/// dispatch gate) and a `(dtype_tag, head_dim) -> Option<&'static str>`
/// symbol lookup. `kernel_identity.rs` re-exports both.
fn write_steel_paged_kernels_rs(out_dir: &std::path::Path) {
    let mut s = String::from(
        "// SPDX-License-Identifier: Apache-2.0\n\
         // Auto-generated by `ferrite-metal-kernels/build.rs`.\n\
         // Edit `STEEL_PAGED_HEAD_DIMS` in build.rs, not here.\n\n",
    );
    s.push_str(
        "/// HEAD_DIMs for which `attention_steel_paged.metal` has a\n\
         /// kernel instantiation (BD template arg). Single source of\n\
         /// truth shared with the Metal compile via the generated\n\
         /// `attention_steel_paged_instantiations.h`.\n",
    );
    s.push_str("pub const STEEL_PAGED_HEAD_DIMS: &[u32] = &[");
    for (i, &bd) in STEEL_PAGED_HEAD_DIMS.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&bd.to_string());
    }
    s.push_str("];\n\n");

    s.push_str(
        "/// MSL symbol for `attention_steel_paged_<dtype>_bq32_bk16_bd<head_dim>_wm4_wn1_bs16`.\n\
         /// Returns `None` when the combo isn't instantiated; caller\n\
         /// must fall back to the per-token SDPA path.\n\
         pub fn steel_paged_symbol(dtype_tag: &str, head_dim: u32) -> Option<&'static str> {\n\
         \x20   match (dtype_tag, head_dim) {\n",
    );
    for &bd in STEEL_PAGED_HEAD_DIMS {
        for &(tag, _ty) in STEEL_PAGED_DTYPES {
            let sym = format!(
                "attention_steel_paged_{tag}_bq32_bk16_bd{bd}_wm4_wn1_bs16"
            );
            s.push_str(&format!(
                "        ({tag:?}, {bd}) => Some({sym:?}),\n"
            ));
        }
    }
    s.push_str("        _ => None,\n    }\n}\n");

    std::fs::write(
        out_dir.join("steel_paged_kernels_generated.rs"),
        s,
    )
    .expect("write steel_paged_kernels_generated.rs");
}
