//! Test: inject SiLU into CUTLASS GEMM epilogue via PTX rewriting.
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_gemm_epilogue -- --nocapture

#![cfg(feature = "cuda")]

use ptx_fusion_macros::{extract_entry, inject_silu_epilogue};

// The first bf16 CUTLASS GEMM entry (16x64x128 tile, FastAccum)
// Using the shortest unique-enough prefix — extract_entry takes the first match
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

#[test]
fn silu_injected_ptx_passes_ptxas() {
    let silu_count = CUTLASS_GEMM_SILU_PTX
        .matches("FERRITE: inject SiLU")
        .count();
    println!("SiLU injected at {} sites", silu_count);
    assert!(silu_count > 0);
    assert!(CUTLASS_GEMM_SILU_PTX.contains("%f_silu"));

    // Validate with ptxas
    let path = "/tmp/cutlass_gemm_with_silu.ptx";
    std::fs::write(path, CUTLASS_GEMM_SILU_PTX).unwrap();

    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines().take(10) {
            println!("ptxas: {line}");
        }
        panic!("ptxas FAILED on SiLU-injected CUTLASS GEMM");
    }

    println!(
        "PASS: SiLU-injected CUTLASS GEMM passes ptxas ({} lines, {} SiLU sites)",
        CUTLASS_GEMM_SILU_PTX.lines().count(),
        silu_count
    );
}
