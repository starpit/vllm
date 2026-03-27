//! Test: escape perimeter extraction across all kernel types.
//!
//! Validates that the perimeter model correctly captures:
//! 1. Hand-written PTX: ld.global/st.global
//! 2. nvcc PTX: ld.global.nc with cvta chains
//! 3. nvcc PTX with struct params: CUTLASS-style [%rd1+offset]
//! 4. CUTLASS PTX: cp.async.cg.shared.global as async input perimeter
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_cutlass_bf16 -- --nocapture

#![cfg(feature = "cuda")]

use ptx_fusion::ParamRole;
use ptx_fusion_macros::{
    analyze_kernel_as, delete_cutlass_a_loads, fuse_norm_gemm_silu, fuse_rms_norm_cutlass,
    replace_cutlass_a_loads, replace_perimeter_macro,
};

// ── Protocol extraction for every kernel type ──

// Hand-written: simple ld.global/st.global
analyze_kernel_as!("kernels/rms_norm.ptx", RMS_NORM_HANDWRITTEN);

// nvcc: ld.global.nc with cvta + add chains
analyze_kernel_as!("kernels/rms_norm_real.ptx", RMS_NORM_NVCC);

// nvcc row GEMM: ld.global.nc, mul.wide address chains
analyze_kernel_as!("kernels/gemm_row_f32.ptx", GEMM_ROW);

// CUTLASS bf16: cp.async + mma.sync + struct params
analyze_kernel_as!("kernels/cutlass_gemm_bf16_sm89.ptx", CUTLASS_BF16);

// CUTLASS bf16 with A-matrix cp.async deleted (prologue injection target)
delete_cutlass_a_loads!(
    "kernels/cutlass_gemm_bf16_sm89.ptx",
    "Gemm",
    CUTLASS_BF16_NO_A_LOADS
);

// CUTLASS bf16 with rms_norm fused into A-matrix loads
fuse_rms_norm_cutlass!(
    "kernels/cutlass_gemm_bf16_sm89.ptx",
    "Gemm",
    "fused_rms_norm_cutlass_gemm",
    FUSED_RMS_NORM_CUTLASS_PTX
);

// CUTLASS bf16 with rms_norm prologue + SiLU epilogue (both fused)
fuse_norm_gemm_silu!(
    "kernels/cutlass_gemm_bf16_sm89.ptx",
    "Gemm",
    "fused_norm_gemm_silu",
    FUSED_NORM_GEMM_SILU_PTX
);

// CUTLASS bf16 with A-matrix cp.async replaced by explicit ld.global + st.shared
replace_cutlass_a_loads!(
    "kernels/cutlass_gemm_bf16_sm89.ptx",
    "Gemm",
    CUTLASS_BF16_EXPLICIT_A
);

// CUTLASS FP8: multi-entry file — extract single entry, then analyze.
// analyze_kernel_as! rejects multi-entry PTX (must extract first).
ptx_fusion_macros::extract_entry!(
    "kernels/cutlass_gemm_sm89.ptx",
    "GemmShapeILi16ELi64ELi128",
    CUTLASS_FP8_PTX
);

#[test]
fn perimeter_hand_written() {
    println!("=== Hand-written rms_norm ===");
    RMS_NORM_HANDWRITTEN.display();

    assert!(!RMS_NORM_HANDWRITTEN.global_loads.is_empty());
    assert!(!RMS_NORM_HANDWRITTEN.global_stores.is_empty());
    assert_eq!(RMS_NORM_HANDWRITTEN.async_loads.len(), 0);
    assert!(!RMS_NORM_HANDWRITTEN.has_mma);

    println!(
        "PASS: {} ld.global, {} st.global, 0 cp.async",
        RMS_NORM_HANDWRITTEN.global_loads.len(),
        RMS_NORM_HANDWRITTEN.global_stores.len()
    );
}

#[test]
fn perimeter_nvcc() {
    println!("=== nvcc rms_norm ===");
    RMS_NORM_NVCC.display();

    assert!(!RMS_NORM_NVCC.global_loads.is_empty());
    assert!(!RMS_NORM_NVCC.global_stores.is_empty());
    assert_eq!(RMS_NORM_NVCC.async_loads.len(), 0);
    assert!(!RMS_NORM_NVCC.has_mma);

    println!(
        "PASS: {} ld.global, {} st.global, 0 cp.async",
        RMS_NORM_NVCC.global_loads.len(),
        RMS_NORM_NVCC.global_stores.len()
    );
}

#[test]
fn perimeter_row_gemm() {
    println!("=== nvcc row GEMM ===");
    GEMM_ROW.display();

    assert!(!GEMM_ROW.global_loads.is_empty());
    assert!(!GEMM_ROW.global_stores.is_empty());
    assert_eq!(GEMM_ROW.async_loads.len(), 0);
    assert!(!GEMM_ROW.has_mma);

    println!(
        "PASS: {} ld.global, {} st.global, 0 cp.async, no MMA",
        GEMM_ROW.global_loads.len(),
        GEMM_ROW.global_stores.len()
    );
}

#[test]
fn perimeter_cutlass_bf16() {
    println!("=== CUTLASS bf16 GEMM ===");
    CUTLASS_BF16.display();

    // CUTLASS GEMM input perimeter is cp.async, not ld.global
    assert!(
        !CUTLASS_BF16.async_loads.is_empty(),
        "CUTLASS should have cp.async input perimeter"
    );
    assert!(CUTLASS_BF16.has_mma, "CUTLASS should have MMA");

    // Async loads trace to exactly 2 distinct params (A and B)
    let mut async_params: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for a in CUTLASS_BF16.async_loads {
        async_params.insert(a.param_name);
    }
    println!("  Async load params: {:?}", async_params);
    assert_eq!(
        async_params.len(),
        2,
        "should trace to exactly 2 params (A and B)"
    );

    // Count per param
    for param in &async_params {
        let count = CUTLASS_BF16
            .async_loads
            .iter()
            .filter(|a| a.param_name == *param)
            .count();
        println!("    {param}: {count} cp.async loads");
        assert!(count > 0);
    }

    // Every async load has valid registers and 16-byte size
    for a in CUTLASS_BF16.async_loads {
        assert!(a.smem_dst.starts_with('%'), "bad SMEM dst: {}", a.smem_dst);
        assert!(a.gmem_src.starts_with('%'), "bad GMEM src: {}", a.gmem_src);
        assert_eq!(a.size_bytes, 16);
    }

    println!(
        "PASS: {} cp.async across 2 params, {} MMA",
        CUTLASS_BF16.async_loads.len(),
        if CUTLASS_BF16.has_mma { "yes" } else { "no" }
    );
}

#[test]
fn perimeter_cutlass_fp8() {
    println!("=== CUTLASS FP8 GEMM (extracted single entry) ===");

    // Verify the extracted entry has cp.async and MMA
    let cp_async_count = CUTLASS_FP8_PTX.matches("cp.async.cg.shared.global").count();
    let mma_count = CUTLASS_FP8_PTX.matches("mma.sync").count();
    let entry_count = CUTLASS_FP8_PTX
        .lines()
        .filter(|l| l.contains(".entry") && l.contains('('))
        .count();

    println!("  Entries: {entry_count}");
    println!("  cp.async: {cp_async_count}");
    println!("  mma.sync: {mma_count}");

    assert_eq!(
        entry_count, 1,
        "should have exactly 1 entry after extraction"
    );
    assert!(cp_async_count > 0, "FP8 CUTLASS should have cp.async");
    assert!(mma_count > 0, "FP8 CUTLASS should have MMA");

    println!("PASS: FP8 CUTLASS extracted entry has {cp_async_count} cp.async, {mma_count} MMA");
}

#[test]
fn delete_a_loads_passes_ptxas() {
    let original_cp = CUTLASS_BF16.async_loads.len();
    let remaining_cp = CUTLASS_BF16_NO_A_LOADS
        .matches("cp.async.cg.shared.global")
        .count();
    let deleted = CUTLASS_BF16_NO_A_LOADS
        .matches("FERRITE: deleted cp.async for A-matrix")
        .count();

    println!("=== CUTLASS bf16 A-load deletion ===");
    println!("  Original cp.async: {original_cp}");
    println!("  Remaining cp.async: {remaining_cp} (B-matrix only)");
    println!("  Deleted (A-matrix): {deleted}");

    assert!(deleted > 0, "should have deleted some cp.async");
    assert!(remaining_cp > 0, "should preserve B-matrix cp.async");
    assert_eq!(
        deleted + remaining_cp,
        original_cp,
        "deleted + remaining should equal original"
    );

    // MMA instructions should be preserved
    let mma_count = CUTLASS_BF16_NO_A_LOADS.matches("mma.sync").count();
    assert!(mma_count > 0, "MMA should be preserved");

    // Validate with ptxas
    let path = "/tmp/cutlass_bf16_no_a_loads.ptx";
    std::fs::write(path, CUTLASS_BF16_NO_A_LOADS).unwrap();

    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines().take(20) {
            println!("ptxas: {line}");
        }
        panic!("ptxas FAILED on A-load-deleted CUTLASS GEMM");
    }

    println!(
        "PASS: A-load-deleted CUTLASS bf16 passes ptxas ({deleted} deleted, {remaining_cp} preserved, {mma_count} MMA)"
    );
}

#[test]
fn replace_a_loads_passes_ptxas() {
    // Verify structure: no cp.async for A, but ld.global + st.shared present
    let explicit_markers = CUTLASS_BF16_EXPLICIT_A
        .matches("FERRITE: explicit ld+st replacing cp.async")
        .count();
    let remaining_cp = CUTLASS_BF16_EXPLICIT_A
        .matches("cp.async.cg.shared.global")
        .count();
    let ld_globals = CUTLASS_BF16_EXPLICIT_A.matches("ld.global.v4.b32").count();
    let st_shareds = CUTLASS_BF16_EXPLICIT_A.matches("st.shared.v4.b32").count();
    let mma_count = CUTLASS_BF16_EXPLICIT_A.matches("mma.sync").count();

    println!("=== CUTLASS bf16 A-load replacement (explicit ld+st) ===");
    println!("  Replaced cp.async: {explicit_markers}");
    println!("  Remaining cp.async: {remaining_cp} (B-matrix only)");
    println!("  ld.global.v4.b32: {ld_globals}");
    println!("  st.shared.v4.b32: {st_shareds}");
    println!("  mma.sync: {mma_count}");

    assert!(
        explicit_markers > 0,
        "should have replaced some cp.async with explicit ld+st"
    );
    assert!(remaining_cp > 0, "should preserve B-matrix cp.async");
    assert_eq!(
        ld_globals, explicit_markers,
        "each replacement should have one ld.global.v4.b32"
    );
    assert_eq!(
        st_shareds, explicit_markers,
        "each replacement should have one st.shared.v4.b32"
    );
    assert!(mma_count > 0, "MMA should be preserved");

    // Validate with ptxas
    let path = "/tmp/cutlass_bf16_explicit_a.ptx";
    std::fs::write(path, CUTLASS_BF16_EXPLICIT_A).unwrap();

    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines().take(30) {
            println!("ptxas: {line}");
        }
        panic!("ptxas FAILED on explicit-A-load CUTLASS GEMM");
    }

    println!(
        "PASS: explicit-A-load CUTLASS bf16 passes ptxas ({explicit_markers} replaced, {remaining_cp} B-preserved, {mma_count} MMA)"
    );
}

#[test]
fn explicit_a_loads_gpu_correctness() {
    // Write both PTX files for the C++ test harness
    let orig_path = "/tmp/cutlass_bf16_original.ptx";
    let mod_path = "/tmp/cutlass_bf16_explicit_a.ptx";

    // Extract original PTX (single entry)
    ptx_fusion_macros::extract_entry!(
        "kernels/cutlass_gemm_bf16_sm89.ptx",
        "Gemm",
        CUTLASS_BF16_ORIGINAL
    );
    std::fs::write(orig_path, CUTLASS_BF16_ORIGINAL).unwrap();
    std::fs::write(mod_path, CUTLASS_BF16_EXPLICIT_A).unwrap();

    // Compile the C++ test harness (if not already compiled)
    let test_bin = "/tmp/cutlass_prologue_test";
    let test_src = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/cutlass_prologue_test.cu"
    );

    let compile = std::process::Command::new("/usr/local/cuda-12.9/bin/nvcc")
        .args([
            "-arch=sm_89",
            "-O2",
            "-std=c++17",
            "-I/home/moosevan/.cache/cutlass/include",
            "-o",
            test_bin,
            test_src,
            "-lcuda",
        ])
        .output()
        .expect("nvcc");

    if !compile.status.success() {
        let stderr = String::from_utf8_lossy(&compile.stderr);
        // Allow warnings, only fail on errors
        if stderr.contains("error") {
            for line in stderr.lines().filter(|l| l.contains("error")).take(10) {
                println!("nvcc: {line}");
            }
            panic!("nvcc compilation FAILED");
        }
    }

    // Run the test
    let run = std::process::Command::new(test_bin)
        .args([orig_path, mod_path])
        .output()
        .expect("test binary");

    let stdout = String::from_utf8_lossy(&run.stdout);
    for line in stdout.lines() {
        println!("  {line}");
    }

    if !run.status.success() {
        let stderr = String::from_utf8_lossy(&run.stderr);
        for line in stderr.lines() {
            println!("  ERR: {line}");
        }
        panic!("GPU correctness test FAILED");
    }

    println!("PASS: explicit A-loads GPU correctness verified");
}

#[test]
fn fused_rms_norm_cutlass_passes_ptxas() {
    // Verify the fused kernel has the right structure
    let has_entry = FUSED_RMS_NORM_CUTLASS_PTX.contains("fused_rms_norm_cutlass_gemm");
    let has_prologue = FUSED_RMS_NORM_CUTLASS_PTX.contains("FERRITE: rms_norm prologue");
    let has_normalized = FUSED_RMS_NORM_CUTLASS_PTX.contains("FERRITE: normalized A-load");
    let has_inv_rms = FUSED_RMS_NORM_CUTLASS_PTX.contains("rsqrt.approx.f32");
    let remaining_cp = FUSED_RMS_NORM_CUTLASS_PTX
        .matches("cp.async.cg.shared.global")
        .count();
    let mma_count = FUSED_RMS_NORM_CUTLASS_PTX.matches("mma.sync").count();
    let normalized_count = FUSED_RMS_NORM_CUTLASS_PTX
        .matches("FERRITE: normalized A-load")
        .count();

    println!("=== Fused rms_norm+CUTLASS GEMM ===");
    println!("  Has fused entry: {has_entry}");
    println!("  Has prologue: {has_prologue}");
    println!("  Has normalized loads: {has_normalized}");
    println!("  Has rsqrt: {has_inv_rms}");
    println!("  Remaining cp.async: {remaining_cp} (B-matrix only)");
    println!("  Normalized A-loads: {normalized_count}");
    println!("  MMA instructions: {mma_count}");

    assert!(has_entry, "should have fused entry point");
    assert!(has_prologue, "should have rms_norm prologue");
    assert!(has_normalized, "should have normalized A-loads");
    assert!(has_inv_rms, "should compute inv_rms via rsqrt");
    assert!(remaining_cp > 0, "should preserve B-matrix cp.async");
    assert_eq!(normalized_count, 6, "should have 6 normalized A-loads");
    assert!(mma_count > 0, "MMA should be preserved");

    // Write to disk and validate with ptxas
    let path = "/tmp/fused_rms_norm_cutlass.ptx";
    std::fs::write(path, FUSED_RMS_NORM_CUTLASS_PTX).unwrap();

    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines().take(30) {
            println!("ptxas: {line}");
        }
        panic!("ptxas FAILED on fused rms_norm+CUTLASS GEMM");
    }

    println!(
        "PASS: fused rms_norm+CUTLASS GEMM passes ptxas ({normalized_count} normalized, {remaining_cp} B-preserved, {mma_count} MMA)"
    );
}

#[test]
fn fused_rms_norm_cutlass_gpu_correctness() {
    // Write fused PTX for the C++ test harness
    let fused_path = "/tmp/fused_rms_norm_cutlass.ptx";
    std::fs::write(fused_path, FUSED_RMS_NORM_CUTLASS_PTX).unwrap();

    // Compile the C++ test harness (separate binary to avoid race with explicit test)
    let test_bin = "/tmp/cutlass_fused_test";
    let test_src = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/cutlass_prologue_test.cu"
    );

    let compile = std::process::Command::new("/usr/local/cuda-12.9/bin/nvcc")
        .args([
            "-arch=sm_89",
            "-O2",
            "-std=c++17",
            "-I/home/moosevan/.cache/cutlass/include",
            "-o",
            test_bin,
            test_src,
            "-lcuda",
        ])
        .output()
        .expect("nvcc");

    if !compile.status.success() {
        let stderr = String::from_utf8_lossy(&compile.stderr);
        if stderr.contains("error") {
            for line in stderr.lines().filter(|l| l.contains("error")).take(10) {
                println!("nvcc: {line}");
            }
            panic!("nvcc compilation FAILED");
        }
    }

    // Run the fused test
    let run = std::process::Command::new(test_bin)
        .args(["--fused", fused_path])
        .output()
        .expect("test binary");

    let stdout = String::from_utf8_lossy(&run.stdout);
    for line in stdout.lines() {
        println!("  {line}");
    }

    if !run.status.success() {
        let stderr = String::from_utf8_lossy(&run.stderr);
        for line in stderr.lines() {
            println!("  ERR: {line}");
        }
        panic!("Fused rms_norm GPU correctness test FAILED");
    }

    println!("PASS: fused rms_norm+CUTLASS GEMM GPU correctness verified");
}

#[test]
fn fused_norm_gemm_silu_passes_ptxas() {
    let has_entry = FUSED_NORM_GEMM_SILU_PTX.contains("fused_norm_gemm_silu");
    let has_prologue = FUSED_NORM_GEMM_SILU_PTX.contains("FERRITE: rms_norm prologue");
    let has_silu = FUSED_NORM_GEMM_SILU_PTX.contains("FERRITE: inject SiLU");
    let has_normalized = FUSED_NORM_GEMM_SILU_PTX
        .matches("FERRITE: normalized A-load")
        .count();
    let silu_sites = FUSED_NORM_GEMM_SILU_PTX
        .matches("FERRITE: inject SiLU")
        .count();
    let remaining_cp = FUSED_NORM_GEMM_SILU_PTX
        .matches("cp.async.cg.shared.global")
        .count();
    let mma_count = FUSED_NORM_GEMM_SILU_PTX.matches("mma.sync").count();

    println!("=== Fused norm+GEMM+SiLU ===");
    println!("  Entry: {has_entry}");
    println!("  Prologue (rms_norm): {has_prologue}");
    println!("  Epilogue (SiLU): {has_silu} ({silu_sites} sites)");
    println!("  Normalized A-loads: {has_normalized}");
    println!("  Remaining cp.async: {remaining_cp} (B-matrix)");
    println!("  MMA: {mma_count}");

    assert!(has_entry);
    assert!(has_prologue);
    assert!(has_silu);
    assert_eq!(has_normalized, 6);
    assert!(silu_sites > 0);
    assert!(remaining_cp > 0);
    assert!(mma_count > 0);

    let path = "/tmp/fused_norm_gemm_silu.ptx";
    std::fs::write(path, FUSED_NORM_GEMM_SILU_PTX).unwrap();

    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in stderr.lines().take(30) {
            println!("ptxas: {line}");
        }
        panic!("ptxas FAILED on norm+GEMM+SiLU");
    }

    println!(
        "PASS: norm+GEMM+SiLU passes ptxas ({has_normalized} norm, {silu_sites} SiLU, {remaining_cp} B, {mma_count} MMA)"
    );
}

#[test]
fn fused_norm_gemm_silu_gpu_correctness() {
    let fused_path = "/tmp/fused_norm_gemm_silu.ptx";
    std::fs::write(fused_path, FUSED_NORM_GEMM_SILU_PTX).unwrap();

    let test_bin = "/tmp/cutlass_silu_test";
    let test_src = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/cutlass_prologue_test.cu"
    );

    let compile = std::process::Command::new("/usr/local/cuda-12.9/bin/nvcc")
        .args([
            "-arch=sm_89",
            "-O2",
            "-std=c++17",
            "-I/home/moosevan/.cache/cutlass/include",
            "-o",
            test_bin,
            test_src,
            "-lcuda",
        ])
        .output()
        .expect("nvcc");

    if !compile.status.success() {
        let stderr = String::from_utf8_lossy(&compile.stderr);
        if stderr.contains("error") {
            for line in stderr.lines().filter(|l| l.contains("error")).take(10) {
                println!("nvcc: {line}");
            }
            panic!("nvcc compilation FAILED");
        }
    }

    let run = std::process::Command::new(test_bin)
        .args(["--fused-silu", fused_path])
        .output()
        .expect("test binary");

    let stdout = String::from_utf8_lossy(&run.stdout);
    for line in stdout.lines() {
        println!("  {line}");
    }

    if !run.status.success() {
        let stderr = String::from_utf8_lossy(&run.stderr);
        for line in stderr.lines() {
            println!("  ERR: {line}");
        }
        panic!("norm+GEMM+SiLU GPU correctness FAILED");
    }

    println!("PASS: norm+GEMM+SiLU GPU correctness verified");
}

#[test]
fn classify_cutlass_bf16_params() {
    println!("=== CUTLASS bf16 param classification ===");
    CUTLASS_BF16.display();

    let cp = CUTLASS_BF16.classified_params;
    assert!(
        !cp.is_empty(),
        "CUTLASS bf16 should have classified param fields"
    );

    println!("\nClassified fields:");
    for p in cp {
        let role = match p.role {
            ParamRole::Pointer => "Pointer",
            ParamRole::Stride => "Stride",
            ParamRole::Dimension => "Dimension",
            ParamRole::Scalar => "Scalar",
            ParamRole::Derived => "Derived",
        };
        println!("  offset {:4}: {} → {}", p.offset, p.ptx_type, role);
    }

    // Helper to find a field by offset
    let role_at = |offset: i64| -> ParamRole {
        cp.iter()
            .find(|p| p.offset == offset)
            .unwrap_or_else(|| panic!("no field at offset {offset}"))
            .role
    };

    // ── Key verifications from the plan ──

    // Offset 32 (ld.param.u64 %rd51, [%rd1+8]; actual=24+8=32):
    // Used in mul.lo.s64 → Stride (leading dimension / lda)
    assert_eq!(
        role_at(32),
        ParamRole::Stride,
        "offset 32 should be Stride (lda, feeds mul.lo.s64)"
    );

    // Offset 64 (ld.param.u64 %rd7, [%rd1+40]; actual=24+40=64):
    // Feeds add.s64 → cp.async address; the other add operand traces to mul → Pointer
    assert_eq!(
        role_at(64),
        ParamRole::Pointer,
        "offset 64 should be Pointer (A matrix base ptr)"
    );

    // Offset 40 (ld.param.u64 %rd2, [%rd1+16]; actual=24+16=40):
    // Added to already-complete addresses — inc_strided → Derived
    assert_eq!(
        role_at(40),
        ParamRole::Derived,
        "offset 40 should be Derived (inc_strided)"
    );

    // Offset 48 (ld.param.u64 %rd3, [%rd1+24]; actual=24+24=48):
    // inc_advance or similar → Derived
    assert_eq!(
        role_at(48),
        ParamRole::Derived,
        "offset 48 should be Derived (inc_advance)"
    );

    // Offset 56 (ld.param.u64 %rd4, [%rd1+32]; actual=24+32=56):
    // Another increment → Derived
    assert_eq!(
        role_at(56),
        ParamRole::Derived,
        "offset 56 should be Derived"
    );

    // Offset 12 (grid_tiled_shape.n) → used in setp → Dimension
    assert_eq!(
        role_at(12),
        ParamRole::Dimension,
        "offset 12 should be Dimension (grid_tiled_shape.n)"
    );

    // Offset 16 (grid_tiled_shape.k or similar) → used in setp → Dimension
    assert_eq!(
        role_at(16),
        ParamRole::Dimension,
        "offset 16 should be Dimension (grid_tiled_shape.k)"
    );

    // f32 params should be Scalar (alpha/beta at epilogue offsets)
    let scalar_count = cp.iter().filter(|p| p.role == ParamRole::Scalar).count();
    assert!(
        scalar_count >= 1,
        "should have at least 1 Scalar field (alpha/beta)"
    );

    // Sanity: count by role
    let pointer_count = cp.iter().filter(|p| p.role == ParamRole::Pointer).count();
    let stride_count = cp.iter().filter(|p| p.role == ParamRole::Stride).count();
    let dim_count = cp.iter().filter(|p| p.role == ParamRole::Dimension).count();
    let derived_count = cp.iter().filter(|p| p.role == ParamRole::Derived).count();

    println!("\nRole counts:");
    println!("  Pointer:   {pointer_count}");
    println!("  Stride:    {stride_count}");
    println!("  Dimension: {dim_count}");
    println!("  Scalar:    {scalar_count}");
    println!("  Derived:   {derived_count}");

    // A CUTLASS GEMM should have multiple pointers (A, B, C, D ptrs + epilogue ptrs)
    assert!(
        pointer_count >= 2,
        "should have at least 2 Pointer fields (A and B)"
    );

    // Should have stride fields (lda, ldb, ldc, ldd)
    assert!(stride_count >= 2, "should have at least 2 Stride fields");

    println!("PASS: CUTLASS bf16 param classification verified");
}

// ── Perimeter replacement (Phase 2) ──

// Rewrite the CUTLASS 64x64x32 kernel to use flat params
replace_perimeter_macro!(
    "kernels/cutlass_gemm_bf16_sm89.ptx",
    "kernels/cutlass_bf16_64x64x32_sm89.derivations.json",
    "ferrite_gemm_64x64x32",
    FLAT_GEMM_64x64x32
);

#[test]
fn flat_param_gemm_passes_ptxas() {
    println!("=== Flat-param CUTLASS GEMM (perimeter replacement) ===");

    // Structural sanity
    assert!(
        FLAT_GEMM_64x64x32.contains(".entry ferrite_gemm_64x64x32("),
        "should have ferrite entry name"
    );
    assert!(
        FLAT_GEMM_64x64x32.contains("ferrite_params[88]"),
        "should have flat param declaration"
    );
    assert!(
        !FLAT_GEMM_64x64x32
            .lines()
            .any(|l| l.trim().starts_with("ld.param") && l.contains("_param_0")),
        "should have no old param references in ld.param"
    );

    // MMA preserved
    let mma_count = FLAT_GEMM_64x64x32.matches("mma.sync").count();
    assert!(mma_count > 0, "MMA should be preserved");

    // Validate with ptxas
    let path = "/tmp/flat_gemm_64x64x32.ptx";
    std::fs::write(path, FLAT_GEMM_64x64x32).unwrap();

    let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
        .args(["-arch=sm_89", path])
        .output()
        .expect("ptxas");

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        // Print first few errors for debugging
        for line in stderr.lines().take(20) {
            println!("ptxas: {line}");
        }

        // Also dump the PTX around the first error for debugging
        if let Some(err_line) = stderr.lines().find(|l| l.contains("error")) {
            println!("\nFirst error: {err_line}");
        }

        panic!("ptxas FAILED on flat-param GEMM");
    }

    let cp_async = FLAT_GEMM_64x64x32.matches("cp.async").count();
    let replaced = FLAT_GEMM_64x64x32
        .lines()
        .filter(|l| l.contains("[ferrite] replaced"))
        .count();

    println!(
        "PASS: flat-param GEMM passes ptxas ({replaced} replaced, {mma_count} MMA, {cp_async} cp.async)"
    );
}
