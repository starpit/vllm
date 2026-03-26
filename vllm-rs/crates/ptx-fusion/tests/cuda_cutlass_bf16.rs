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

use ptx_fusion_macros::analyze_kernel_as;

// ── Protocol extraction for every kernel type ──

// Hand-written: simple ld.global/st.global
analyze_kernel_as!("kernels/rms_norm.ptx", RMS_NORM_HANDWRITTEN);

// nvcc: ld.global.nc with cvta + add chains
analyze_kernel_as!("kernels/rms_norm_real.ptx", RMS_NORM_NVCC);

// nvcc row GEMM: ld.global.nc, mul.wide address chains
analyze_kernel_as!("kernels/gemm_row_f32.ptx", GEMM_ROW);

// CUTLASS bf16: cp.async + mma.sync + struct params
analyze_kernel_as!("kernels/cutlass_gemm_bf16_sm89.ptx", CUTLASS_BF16);

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
