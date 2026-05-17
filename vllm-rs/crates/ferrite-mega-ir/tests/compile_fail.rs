// SPDX-License-Identifier: Apache-2.0
//! Compile-fail test driver for const-generic substrate-proof
//! rejection.
//!
//! Each `tests/compile-fail/*.rs` constructs a primitive with const
//! args that violate a substrate invariant. We invoke `rustc` on
//! each file with the ferrite-mega-ir extern, and assert the
//! compiler exits non-zero AND its stderr mentions the expected
//! substrate-proof error message.
//!
//! We DON'T use `trybuild` because trybuild has a known issue with
//! post-monomorphization E0080 const-eval errors (the primary span
//! lives inside `core::panic`, which trybuild filters out as
//! "not in user file"). The hand-rolled driver below works around
//! that by checking exit code + substring match in stderr.
//!
//! Run with `cargo test -p ferrite-mega-ir --test compile_fail`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn target_dir() -> PathBuf {
    let mut d = manifest_dir();
    d.pop();
    d.pop();
    d.push("target");
    d.push("compile-fail");
    d
}

/// Find the latest `libferrite_mega_ir-*.rlib` in the workspace
/// target dir. The trybuild-style bin invocation links against the
/// library; the hand-rolled approach passes `--extern` to `rustc`
/// directly so we don't need a `Cargo.toml`.
fn find_lib_artifact() -> PathBuf {
    let mut workspace_target = manifest_dir();
    workspace_target.pop();
    workspace_target.pop();
    workspace_target.push("target");
    workspace_target.push("debug");
    workspace_target.push("deps");
    let mut latest: Option<(PathBuf, std::time::SystemTime)> = None;
    if let Ok(entries) = std::fs::read_dir(&workspace_target) {
        for e in entries.flatten() {
            let p = e.path();
            let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !name.starts_with("libferrite_mega_ir-") || !name.ends_with(".rlib") {
                continue;
            }
            let Ok(meta) = e.metadata() else { continue };
            let Ok(mtime) = meta.modified() else { continue };
            if latest.as_ref().is_none_or(|(_, t)| mtime > *t) {
                latest = Some((p, mtime));
            }
        }
    }
    latest
        .expect("no libferrite_mega_ir-*.rlib found — run `cargo build -p ferrite-mega-ir` first")
        .0
}

/// Invoke `rustc` on `src_path` with the appropriate `--extern`
/// flag pointing at the latest ferrite-mega-ir rlib. Returns
/// `(exit_status_success, stderr)`.
fn rustc_compile(src_path: &Path, output_bin: &Path) -> (bool, String) {
    let lib = find_lib_artifact();
    let deps_dir = lib.parent().expect("rlib has parent");
    let mut workspace_target = manifest_dir();
    workspace_target.pop();
    workspace_target.pop();
    workspace_target.push("target");
    workspace_target.push("debug");
    workspace_target.push("deps");
    let out = Command::new("rustc")
        .arg("--edition=2024")
        .arg("--crate-type=bin")
        .arg("-o")
        .arg(output_bin)
        .arg("--extern")
        .arg(format!("ferrite_mega_ir={}", lib.display()))
        .arg("-L")
        .arg(format!("dependency={}", deps_dir.display()))
        .arg(src_path)
        .output()
        .expect("failed to run rustc");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    (out.status.success(), stderr)
}

struct Case {
    rs: &'static str,
    /// Substring expected in stderr — the rustc panic message that
    /// our `const { assert!(...) }` block emits.
    expect_msg: &'static str,
}

const CASES: &[Case] = &[
    Case {
        rs: "tests/compile-fail/page_id_out_of_bounds.rs",
        expect_msg: "PageId: ID out of bounds",
    },
    Case {
        rs: "tests/compile-fail/page_out_of_bounds.rs",
        expect_msg: "Page: ID out of bounds",
    },
    Case {
        rs: "tests/compile-fail/scratch_out_of_budget.rs",
        expect_msg: "ScratchRegion: OFFSET+BYTES out of substrate scratch budget",
    },
    Case {
        rs: "tests/compile-fail/scratch_overlap.rs",
        expect_msg: "ScratchRegion overlap within scope",
    },
    Case {
        rs: "tests/compile-fail/mbarrier_phase_parity_mismatch.rs",
        expect_msg: "MbarrierPhase: parity mismatch",
    },
    Case {
        rs: "tests/compile-fail/iter_count_zero.rs",
        expect_msg: "IterCount: ITERS must be > 0",
    },
    Case {
        rs: "tests/compile-fail/layer_index_out_of_range.rs",
        expect_msg: "LayerIndex: LAYER out of range",
    },
    Case {
        rs: "tests/compile-fail/edge_id_out_of_bounds.rs",
        expect_msg: "EdgeId: IDX out of bounds",
    },
    Case {
        rs: "tests/compile-fail/expected_count_zero.rs",
        expect_msg: "ExpectedCount: COUNT must be > 0",
    },
    Case {
        rs: "tests/compile-fail/rms_norm_in_weight_alias.rs",
        expect_msg: "RmsNorm: IN_ID and WEIGHT_ID alias",
    },
    Case {
        rs: "tests/compile-fail/rms_norm_phase_parity.rs",
        expect_msg: "RmsNorm: CONSUMER_PHASE parity mismatch",
    },
    Case {
        rs: "tests/compile-fail/substrate_budget_zero_pages.rs",
        expect_msg: "SubstrateBudget: NUM_PAGES must be > 0",
    },
    Case {
        rs: "tests/compile-fail/sliding_window_zero.rs",
        expect_msg: "SlidingWindow: WINDOW must be > 0",
    },
    Case {
        rs: "tests/compile-fail/matmul_shape_zero.rs",
        expect_msg: "MatmulShape: N must be > 0",
    },
    // Sealed-witness type-checks for bar.sync IDs. These trip
    // `error[E0277]: the trait bound ...` at type-check, NOT
    // `assert!()` at monomorphization. The compile-fail driver
    // greps stderr; trait-bound errors mention the unsatisfied
    // trait name.
    Case {
        rs: "tests/compile-fail/rms_norm_bar_reduce_zero.rs",
        expect_msg: "IsValidBarSyncId",
    },
    Case {
        rs: "tests/compile-fail/rms_norm_bar_publish_oob.rs",
        expect_msg: "IsValidBarSyncId",
    },
    Case {
        rs: "tests/compile-fail/rms_norm_bar_alias.rs",
        expect_msg: "IsDistinctBarPair",
    },
];

#[test]
fn compile_fail_substrate_proofs() {
    let out_dir = target_dir();
    std::fs::create_dir_all(&out_dir).expect("create target/compile-fail");

    let mut failures: Vec<String> = Vec::new();
    for case in CASES {
        let src = manifest_dir().join(case.rs);
        let out_bin = out_dir.join(
            Path::new(case.rs)
                .file_stem()
                .expect("file_stem")
                .to_string_lossy()
                .into_owned(),
        );
        let (compiled, stderr) = rustc_compile(&src, &out_bin);
        if compiled {
            failures.push(format!(
                "[{}] expected compile failure but rustc succeeded",
                case.rs
            ));
        } else if !stderr.contains(case.expect_msg) {
            failures.push(format!(
                "[{}] compile failed but stderr did not contain {:?}\n--- stderr ---\n{}",
                case.rs, case.expect_msg, stderr
            ));
        }
    }
    if !failures.is_empty() {
        let n = failures.len();
        let total = CASES.len();
        panic!(
            "{n} of {total} compile-fail tests failed:\n\n{}",
            failures.join("\n\n")
        );
    }
}
