// SPDX-License-Identifier: Apache-2.0
//! S16 phase 3 — nvcc validation of `cuda_emit` output.
//!
//! These tests render `.cu` source via the per-MegaNode `render_*`
//! fns + `cuda_emit::render_canonical`, write the source to a temp
//! directory, and shell out to `nvcc` with the same flag set
//! `ferrite-cuda-builder/build.rs` uses for the production
//! megakernel build:
//!
//! ```text
//! nvcc -arch=sm_90a -std=c++20
//!      -O3 --use_fast_math --expt-extended-lambda
//!      --expt-relaxed-constexpr -DNDEBUG -DKITTENS_HOPPER
//!      -Xcompiler=-fPIC -Xcompiler=-fno-strict-aliasing
//!      -I <workspace>/crates/ferrite-kernels/csrc/tk
//!      -I <workspace>/third_party/thunderkittens/include
//!      --ptx <emitted>.cu -o <emitted>.ptx
//! ```
//!
//! All tests are `#[ignore]` because nvcc/Hopper aren't available
//! on the macOS dev box; run on the H100 pod via
//! `cargo test -p ferrite-megakernel --test nvcc_compile -- --ignored --nocapture`.
//!
//! Rules: per [[feedback-mega-ir-pure-transcription]] and
//! [[feedback-tk-2-0-only]], any binding fix flagged by nvcc must
//! land in `cuda_emit/tk20.rs` (cite TK 2.0 header line) — we do
//! not patch the emitted `.cu` after the fact.
//!
//! Per [[feedback-end-to-end-compile-time-proofs]], shape mismatches
//! are Rust compile errors via const generics — these tests catch
//! the residue (TK API surface, lambda capture, dtype casts) that
//! the type system can't yet rule out.
//!
//! DOD: every test in this file passes on the pod. Once green,
//! these are the structural floor — any `cuda_emit` change that
//! breaks them is a regression.

use std::path::PathBuf;
use std::process::Command;

use ferrite_megakernel::cuda_emit::{render, render_canonical, LaunchTier};
use ferrite_megakernel::ir::TapeBudget;

/// Resolve the workspace root (`vllm-rs/`) from `CARGO_MANIFEST_DIR`,
/// which Cargo points at `<root>/crates/ferrite-megakernel`.
fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent() // crates/
        .and_then(|p| p.parent()) // <root>
        .expect("CARGO_MANIFEST_DIR has at least two ancestors")
        .to_path_buf()
}

/// Pipe `source` through nvcc with the production megakernel flag
/// set; return `Ok(())` on a clean compile, `Err(stderr)` otherwise.
///
/// `name` is used as the temp `.cu` file stem so failure messages
/// point at a stable path the developer can re-run nvcc against
/// manually.
fn try_nvcc(source: &str, name: &str) -> Result<(), String> {
    let root = workspace_root();
    let tk_include = root.join("third_party/thunderkittens/include");
    let ferrite_include = root.join("crates/ferrite-kernels/csrc/tk");

    if !tk_include.is_dir() {
        return Err(format!(
            "TK 2.0 include dir missing: {}",
            tk_include.display()
        ));
    }
    if !ferrite_include.is_dir() {
        return Err(format!(
            "ferrite-tk include dir missing: {}",
            ferrite_include.display()
        ));
    }

    let out_dir = std::env::temp_dir().join("ferrite_megakernel_nvcc_check");
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("create_dir_all({}): {e}", out_dir.display()))?;
    let cu_path = out_dir.join(format!("{name}.cu"));
    let ptx_path = out_dir.join(format!("{name}.ptx"));
    std::fs::write(&cu_path, source).map_err(|e| format!("write {}: {e}", cu_path.display()))?;

    let output = Command::new("nvcc")
        .args([
            "-arch=sm_90a",
            "-std=c++20",
            "-O3",
            "--use_fast_math",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-DNDEBUG",
            "-DKITTENS_HOPPER",
            "-Xcompiler=-fPIC",
            "-Xcompiler=-fno-strict-aliasing",
        ])
        .arg("-I")
        .arg(&ferrite_include)
        .arg("-I")
        .arg(&tk_include)
        .arg("--ptx")
        .arg(&cu_path)
        .arg("-o")
        .arg(&ptx_path)
        .output()
        .map_err(|e| format!("spawn nvcc: {e}"))?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    Err(format!(
        "nvcc failed for {} (status {}):\n--- stderr ---\n{stderr}\n--- stdout ---\n{stdout}\n--- source at {} ---",
        cu_path.display(),
        output.status,
        cu_path.display(),
    ))
}

fn budget_attn() -> TapeBudget {
    TapeBudget {
        num_pages: 8,
        num_consumer_warps: 8,
        page_size: 32_768,
        scratch_bytes: 65_536,
        num_edges: 4,
        num_layers: 16,
    }
}

/// AttentionViaCache — full FA-2 algorithm body. llama-3.2-1b-ish
/// const-generic shape (M=16, HEAD_DIM=64, NUM_Q_HEADS=32,
/// NUM_KV_HEADS=8, BLOCK_SIZE=16, NCW=8, NUM_LAYERS=16).
#[test]
#[ignore = "requires nvcc + sm_90a — run on the H100 pod"]
fn nvcc_attention_via_cache() {
    let bodies = vec![render::render_attention_via_cache::<
        16, 64, 32, 8, 16, 256, 8, 16, 1,
    >(
        /*q_in_page_id=*/ 0,
        /*attn_out_page_id=*/ 1,
        /*consumer_phase=*/ 0,
        /*storer_phase=*/ 1,
        /*layer=*/ 5,
        /*q_in_act_slot=*/ 0,
        /*attn_out_act_slot=*/ 1,
        /*score_offset=*/ 0,
        /*pv_offset=*/ 4096,
        /*k_smem_offset=*/ 8192,
        /*v_smem_offset=*/ 24576,
        /*attn_scale=*/ 0.125_f32,
        /*attn_softcap=*/ 0.0_f32,
        /*interleaved=*/ false,
    )];
    let cu = render_canonical("nvcc_attn", &budget_attn(), LaunchTier::Attn, &bodies);
    assert!(cu.skipped_variants.is_empty(), "skipped: {:?}", cu.skipped_variants);
    if let Err(e) = try_nvcc(&cu.source, "attention_via_cache") {
        panic!("{e}");
    }
}

/// SlidingAttentionViaCache — same algorithm with the sliding-window
/// mask. Tests the extra `warp_apply_f32_rt_lambda` arm.
#[test]
#[ignore = "requires nvcc + sm_90a — run on the H100 pod"]
fn nvcc_sliding_attention_via_cache() {
    let bodies = vec![render::render_sliding_attention_via_cache::<
        16, 64, 32, 8, 16, 256, 8, 16, 1,
    >(
        0, 1, 0, 1, 5, 0, 1,
        0, 4096, 8192, 24576,
        0.125_f32, 0.0_f32, false,
        /*sliding_window=*/ 4096,
    )];
    let cu = render_canonical(
        "nvcc_sliding_attn",
        &budget_attn(),
        LaunchTier::Attn,
        &bodies,
    );
    assert!(cu.skipped_variants.is_empty(), "skipped: {:?}", cu.skipped_variants);
    if let Err(e) = try_nvcc(&cu.source, "sliding_attention_via_cache") {
        panic!("{e}");
    }
}

/// AttentionViaCache with a non-zero tanh softcap. Tests the
/// `softcap * tanhf(att / softcap)` lambda arm in addition to the
/// base FA-2 body.
#[test]
#[ignore = "requires nvcc + sm_90a — run on the H100 pod"]
fn nvcc_attention_via_cache_softcap() {
    let bodies = vec![render::render_attention_via_cache::<
        16, 64, 32, 8, 16, 256, 8, 16, 1,
    >(
        0, 1, 0, 1, 5, 0, 1,
        0, 4096, 8192, 24576,
        0.125_f32, /*attn_softcap=*/ 30.0_f32, false,
    )];
    let cu = render_canonical("nvcc_attn_softcap", &budget_attn(), LaunchTier::Attn, &bodies);
    assert!(cu.skipped_variants.is_empty(), "skipped: {:?}", cu.skipped_variants);
    if let Err(e) = try_nvcc(&cu.source, "attention_via_cache_softcap") {
        panic!("{e}");
    }
}
