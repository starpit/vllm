//! Test: inject activations into CUTLASS GEMM epilogue via PTX rewriting.
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_gemm_epilogue -- --nocapture

#![cfg(feature = "cuda")]

use ptx_fusion_macros::{extract_entry, inject_epilogue, inject_silu_epilogue};

// The first bf16 CUTLASS GEMM entry (16x64x128 tile, FastAccum)
// Using the shortest unique-enough prefix -- extract_entry takes the first match
extract_entry!(
    "kernels/cutlass_gemm_sm89.ptx",
    "GemmShapeILi16ELi64ELi128",
    CUTLASS_GEMM_PTX
);

// Same entry with SiLU injected into epilogue
inject_silu_epilogue!(
    "kernels/cutlass_gemm_sm89.ptx",
    "GemmShapeILi16ELi64ELi128",
    CUTLASS_GEMM_SILU_PTX
);

// Same entry with GELU injected
inject_epilogue!(
    "kernels/cutlass_gemm_sm89.ptx",
    "GemmShapeILi16ELi64ELi128",
    Gelu,
    CUTLASS_GEMM_GELU_PTX
);

// Same entry with ReLU injected
inject_epilogue!(
    "kernels/cutlass_gemm_sm89.ptx",
    "GemmShapeILi16ELi64ELi128",
    Relu,
    CUTLASS_GEMM_RELU_PTX
);

#[test]
fn cutlass_gemm_extracts_correctly() {
    let entry_count = CUTLASS_GEMM_PTX
        .lines()
        .filter(|l| l.contains(".entry") && l.contains('('))
        .count();
    println!(
        "Extracted CUTLASS GEMM: {} lines, {} entry(s)",
        CUTLASS_GEMM_PTX.lines().count(),
        entry_count
    );
    assert!(entry_count >= 1);
    assert!(CUTLASS_GEMM_PTX.contains("mma.sync"));
    assert!(CUTLASS_GEMM_PTX.contains("cvt.rn.bf16x2.f32"));
    println!("PASS: CUTLASS GEMM entry extracted");
}

fn validate_ptxas(ptx: &str, path: &str, label: &str) {
    std::fs::write(path, ptx).unwrap();
    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines().take(10) {
            println!("ptxas: {line}");
        }
        panic!("ptxas FAILED on {label}");
    }
}

#[test]
fn silu_injected_ptx_passes_ptxas() {
    let count = CUTLASS_GEMM_SILU_PTX
        .matches("FERRITE: inject SiLU")
        .count();
    println!("SiLU injected at {} sites", count);
    assert!(count > 0);
    assert!(CUTLASS_GEMM_SILU_PTX.contains("%f_act"));

    validate_ptxas(
        CUTLASS_GEMM_SILU_PTX,
        "/tmp/cutlass_gemm_with_silu.ptx",
        "SiLU-injected CUTLASS GEMM",
    );

    println!(
        "PASS: SiLU-injected CUTLASS GEMM passes ptxas ({} lines, {} sites)",
        CUTLASS_GEMM_SILU_PTX.lines().count(),
        count
    );
}

#[test]
fn gelu_injected_ptx_passes_ptxas() {
    let count = CUTLASS_GEMM_GELU_PTX
        .matches("FERRITE: inject GELU")
        .count();
    println!("GELU injected at {} sites", count);
    assert!(count > 0);
    assert!(CUTLASS_GEMM_GELU_PTX.contains("%f_act<8>"));

    validate_ptxas(
        CUTLASS_GEMM_GELU_PTX,
        "/tmp/cutlass_gemm_with_gelu.ptx",
        "GELU-injected CUTLASS GEMM",
    );

    println!(
        "PASS: GELU-injected CUTLASS GEMM passes ptxas ({} lines, {} sites)",
        CUTLASS_GEMM_GELU_PTX.lines().count(),
        count
    );
}

#[test]
fn relu_injected_ptx_passes_ptxas() {
    let count = CUTLASS_GEMM_RELU_PTX
        .matches("FERRITE: inject ReLU")
        .count();
    println!("ReLU injected at {} sites", count);
    assert!(count > 0);
    assert!(
        !CUTLASS_GEMM_RELU_PTX.contains("%f_act"),
        "ReLU should not need scratch registers"
    );
    assert!(CUTLASS_GEMM_RELU_PTX.contains("max.f32"));

    validate_ptxas(
        CUTLASS_GEMM_RELU_PTX,
        "/tmp/cutlass_gemm_with_relu.ptx",
        "ReLU-injected CUTLASS GEMM",
    );

    println!(
        "PASS: ReLU-injected CUTLASS GEMM passes ptxas ({} lines, {} sites)",
        CUTLASS_GEMM_RELU_PTX.lines().count(),
        count
    );
}
