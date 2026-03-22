use ferrite_metal::config::{GpuGeneration, MetalGemmConfig, Precision};
use ferrite_metal::gemm::build_standalone_gemm;

// ═══════════════════════════════════════════════════════════════════
// Config validation — cross-reference with MFA's GEMMKernelDescriptor
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_apple8_tile_sizes() {
    let c = MetalGemmConfig::default_apple8_f16();
    assert_eq!(c.block_m, 48, "MFA apple8 f16: M_group = 48");
    assert_eq!(c.block_n, 48, "MFA apple8 f16: N_group = 48");
    assert_eq!(c.block_k, 32, "MFA apple8 f16: K_group = 32");
    assert!(!c.prefer_async_load, "apple8 does NOT prefer async");
    assert_eq!(c.gpu_gen, GpuGeneration::Apple8);
}

#[test]
fn test_apple9_tile_sizes() {
    let c = MetalGemmConfig::default_apple9_f16();
    assert_eq!(c.block_m, 32, "MFA apple9 f16: M_group = 32");
    assert_eq!(c.block_n, 32, "MFA apple9 f16: N_group = 32");
    assert_eq!(c.block_k, 8, "MFA apple9 f16: K_group = 8");
    assert!(c.prefer_async_load, "apple9 prefers async");
    assert_eq!(c.gpu_gen, GpuGeneration::Apple9);
}

#[test]
fn test_apple9_leading_block_dims() {
    let c = MetalGemmConfig::default_apple9_f16();
    // MFA: apple9 with A=row B=col f16→f32 uses padding (32, 32, 32)
    assert_eq!(c.leading_block_dims, Some([32, 32, 32]));
    assert_eq!(c.leading_block_dim('A'), 32);
    assert_eq!(c.leading_block_dim('B'), 32);
    assert_eq!(c.leading_block_dim('C'), 32);
}

#[test]
fn test_register_dimensions() {
    // With splits=(1,1), register dims = block dims
    let c = MetalGemmConfig::default_apple9_f16();
    assert_eq!(c.register_m(), 32, "registerM = blockM / splits.y = 32/1");
    assert_eq!(c.register_n(), 32, "registerN = blockN / splits.x = 32/1");

    // With splits=(2,2), register dims halve
    let mut c2 = c.clone();
    c2.splits = [2, 2];
    assert_eq!(c2.register_m(), 16, "registerM = 32/2 = 16");
    assert_eq!(c2.register_n(), 16, "registerN = 32/2 = 16");
}

#[test]
fn test_block_bytes_f16() {
    let c = MetalGemmConfig::default_apple9_f16();
    // A: leading=32, trailing=blockM=32, f16=2 bytes → 32*32*2 = 2048
    let a_bytes = c.block_bytes('A');
    assert_eq!(a_bytes, 2048, "A block bytes for 32×32 f16 with leading=32");

    // B: transposed, leading=32, trailing=blockN=32 (transposed flips), f16=2 → 32*32*2 = 2048
    let b_bytes = c.block_bytes('B');
    assert_eq!(
        b_bytes, 2048,
        "B block bytes for transposed 32×32 f16 with leading=32"
    );
}

#[test]
fn test_block_bytes_f16_apple8() {
    let c = MetalGemmConfig::default_apple8_f16();
    // apple8 no explicit leading_block_dims → auto-computed
    // A: not transposed, leading = blockK = 32, trailing = blockM = 48
    let a_bytes = c.block_bytes('A');
    // 32 * 48 * 2 = 3072
    assert_eq!(a_bytes, 3072, "A block bytes for 48×32 f16 auto-leading");

    // B: transposed, leading = blockK = 32, trailing = blockN = 48
    let b_bytes = c.block_bytes('B');
    // 32 * 48 * 2 = 3072
    assert_eq!(
        b_bytes, 3072,
        "B block bytes for 48×32 f16 transposed auto-leading"
    );
}

#[test]
fn test_threadgroup_memory_apple9() {
    let c = MetalGemmConfig::default_apple9_f16();
    let mem = c.threadgroup_memory();
    // A+B = 2048 + 512 = 2560, C = 32*32*4 = 4096 → max(2560, 4096) = 4096
    assert_eq!(mem, 4096, "Threadgroup memory for apple9 32×32 f16→f32");
}

#[test]
fn test_threadgroup_memory_apple8() {
    let c = MetalGemmConfig::default_apple8_f16();
    let mem = c.threadgroup_memory();
    // A+B = 3072 + 3072 = 6144, C = 48*48*4 = 9216 → max(6144, 9216) = 9216
    assert_eq!(mem, 9216, "Threadgroup memory for apple8 48×48 f16→f32");
}

#[test]
fn test_threadgroup_size() {
    let c = MetalGemmConfig::default_apple9_f16();
    assert_eq!(c.threadgroup_size(), 32, "1 simdgroup × 32 threads");

    let mut c2 = c.clone();
    c2.splits = [2, 2];
    assert_eq!(c2.threadgroup_size(), 128, "4 simdgroups × 32 threads");
}

#[test]
fn test_precision_names() {
    assert_eq!(Precision::FP16.msl_name(), "half");
    assert_eq!(Precision::BF16.msl_name(), "bfloat");
    assert_eq!(Precision::FP32.msl_name(), "float");
    assert_eq!(Precision::FP16.bytes(), 2);
    assert_eq!(Precision::FP32.bytes(), 4);
}

#[test]
fn test_transpose_affects_leading_block_dim() {
    // When leading_block_dims is None, the leading dimension depends on transpose
    let mut c = MetalGemmConfig::default_apple8_f16();
    c.leading_block_dims = None;

    // A not transposed: leading = K dimension = block_k
    c.transpose[0] = false;
    assert_eq!(c.leading_block_dim('A'), c.block_k);

    // A transposed: leading = M dimension = block_m
    c.transpose[0] = true;
    assert_eq!(c.leading_block_dim('A'), c.block_m);
}

// ═══════════════════════════════════════════════════════════════════
// MSL generation — structural correctness
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_generates_valid_msl_structure() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);

    // Must have include and namespace
    assert!(
        msl.contains("#include <metal_stdlib>"),
        "Missing metal include"
    );
    assert!(msl.contains("using namespace metal;"), "Missing namespace");

    // Must have kernel declaration
    assert!(
        msl.contains("kernel void gemm("),
        "Missing kernel declaration"
    );

    // Must have buffer bindings
    assert!(msl.contains("[[buffer(0)]]"), "Missing buffer binding 0");
    assert!(msl.contains("[[buffer(1)]]"), "Missing buffer binding 1");
    assert!(msl.contains("[[buffer(2)]]"), "Missing buffer binding 2");

    // Must have simdgroup intrinsics
    assert!(
        msl.contains("simdgroup_matrix_storage"),
        "Missing simdgroup_matrix_storage"
    );
    assert!(msl.contains("morton_order"), "Missing morton_order helper");
    assert!(msl.contains("get_sram"), "Missing get_sram helper");

    // Must have K-loop
    assert!(
        msl.contains("for (uint k = 0; k < K; k += K_group)"),
        "Missing K-loop"
    );

    // Must have threadgroup memory
    assert!(
        msl.contains("threadgroup_block"),
        "Missing threadgroup allocation"
    );
    assert!(msl.contains("threadgroup_barrier"), "Missing barrier");

    // Must have accumulator init and store
    assert!(msl.contains("C_sram"), "Missing C_sram accumulators");
    assert!(msl.contains("store"), "Missing store");
}

#[test]
fn test_apple9_uses_async_copy() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
    assert!(
        msl.contains("simdgroup_event"),
        "apple9 should use async copy"
    );
    assert!(msl.contains("async_copy"), "apple9 should use async_copy");
}

#[test]
fn test_apple8_uses_polyfill_not_hardware_async() {
    let config = MetalGemmConfig::default_apple8_f16();
    let msl = build_standalone_gemm(&config);
    // apple8: polyfill simdgroup_event (loop-based, no AIR intrinsics)
    assert!(
        !msl.contains("__asm(\"air.simdgroup_async_copy"),
        "apple8 should NOT use hardware AIR async copy intrinsics"
    );
    // The polyfill struct `simdgroup_event` is still present but with no-op wait
    assert!(
        msl.contains("simdgroup_event"),
        "Polyfill struct should still be defined"
    );
}

#[test]
fn test_correct_precision_types_in_msl() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
    assert!(msl.contains("half"), "Should use half for f16 operands");
    assert!(
        msl.contains("float"),
        "Should use float for f32 accumulator"
    );
}

#[test]
fn test_constants_match_config() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
    assert!(
        msl.contains("M_group = 32"),
        "M_group should be 32 for apple9"
    );
    assert!(
        msl.contains("N_group = 32"),
        "N_group should be 32 for apple9"
    );
    assert!(
        msl.contains("K_group = 8"),
        "K_group should be 8 for apple9"
    );
}

#[test]
fn test_msl_output_not_empty() {
    for config in [
        MetalGemmConfig::default_apple8_f16(),
        MetalGemmConfig::default_apple9_f16(),
    ] {
        let msl = build_standalone_gemm(&config);
        assert!(
            msl.len() > 500,
            "Generated MSL should be substantial, got {} bytes",
            msl.len()
        );
    }
}

#[test]
fn test_msl_balanced_braces() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
    let opens = msl.chars().filter(|c| *c == '{').count();
    let closes = msl.chars().filter(|c| *c == '}').count();
    assert_eq!(
        opens, closes,
        "Unbalanced braces: {} opens vs {} closes",
        opens, closes
    );
}

#[test]
fn test_no_template_variables_remain() {
    for config in [
        MetalGemmConfig::default_apple9_f16(),
        MetalGemmConfig::default_apple8_f16(),
    ] {
        let msl = build_standalone_gemm(&config);
        assert!(
            !msl.contains("{{"),
            "Unreplaced template variable in {} config: {}",
            if config.prefer_async_load {
                "apple9"
            } else {
                "apple8"
            },
            msl.lines().find(|l| l.contains("{{")).unwrap_or("???")
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// Full MSL structure — validate the complete kernel looks right
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_full_kernel_has_complete_pipeline() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);

    // The kernel should have all pipeline stages in order:
    // 1. Headers (simdgroup_event, simdgroup_matrix_storage)
    // 2. Constants (M_group, N_group, K_group)
    // 3. Utilities (morton_order, get_sram)
    // 4. Kernel signature
    // 5. Thread setup (M_offset, N_offset, morton_offset)
    // 6. Accumulator init (C_sram)
    // 7. K-loop with:
    //    a. Tile copy (async_copy or direct)
    //    b. threadgroup_barrier
    //    c. Fragment loads (A->load, B->load)
    //    d. Multiply (C->multiply)
    //    e. Final barrier
    // 8. Store (C_acc->store)

    let header_pos = msl.find("__METAL_SIMDGROUP_EVENT").unwrap();
    let matrix_pos = msl.find("__METAL_SIMDGROUP_MATRIX_STORAGE").unwrap();
    let const_pos = msl.find("M_group").unwrap();
    let morton_pos = msl.find("morton_order").unwrap();
    let kernel_pos = msl.find("kernel void gemm").unwrap();
    let setup_pos = msl.find("M_offset").unwrap();
    let acc_pos = msl.find("C_sram").unwrap();
    let kloop_pos = msl.find("for (uint k = 0").unwrap();
    let store_pos = msl.find("C_acc->store").unwrap();

    assert!(header_pos < matrix_pos, "Event header before matrix header");
    assert!(matrix_pos < const_pos, "Headers before constants");
    assert!(const_pos < morton_pos, "Constants before utilities");
    assert!(morton_pos < kernel_pos, "Utilities before kernel");
    assert!(kernel_pos < setup_pos, "Kernel before setup");
    assert!(setup_pos < acc_pos, "Setup before accumulators");
    assert!(acc_pos < kloop_pos, "Accumulators before K-loop");
    assert!(kloop_pos < store_pos, "K-loop before store");
}

#[test]
fn test_k_loop_has_correct_inner_structure() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);

    // Within the K-loop, the inner structure should be:
    // 1. Tile copy setup (A_block, B_block pointers)
    // 2. async_copy (apple9) or nothing (apple8)
    // 3. threadgroup_barrier
    // 4. apply_offset for A_block_src, B_block_src
    // 5. Inner k_inner loop (unrolled):
    //    a. A->load (fragment load)
    //    b. B->load (fragment load)
    //    c. C->multiply
    // 6. Final threadgroup_barrier

    // Find the K-loop body
    let kloop_start = msl.find("for (uint k = 0; k < K; k += K_group)").unwrap();
    let kloop_body = &msl[kloop_start..];

    assert!(
        kloop_body.contains("A_block"),
        "K-loop must have A_block pointer"
    );
    assert!(
        kloop_body.contains("B_block"),
        "K-loop must have B_block pointer"
    );
    assert!(
        kloop_body.contains("apply_offset"),
        "K-loop must compute block src offsets"
    );
    assert!(
        kloop_body.contains("for (ushort k_inner"),
        "K-loop must have inner k_inner loop"
    );
    assert!(
        kloop_body.contains("#pragma clang loop unroll(full)"),
        "Inner loop must be unrolled"
    );

    // Fragment loads in correct order within inner loop
    let inner_start = kloop_body.find("for (ushort k_inner").unwrap();
    let inner_body = &kloop_body[inner_start..];
    let a_load = inner_body.find("A->load").unwrap();
    let b_load = inner_body.find("B->load").unwrap();
    let multiply = inner_body.find("C->multiply").unwrap();
    assert!(a_load < b_load, "A load before B load in inner loop");
    assert!(b_load < multiply, "B load before multiply in inner loop");
}

#[test]
fn test_dump_full_msl_apple9() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
    // Print for manual inspection — run with: cargo test test_dump -- --nocapture
    eprintln!(
        "\n=== Generated MSL (apple9 f16, {} bytes) ===\n{}\n=== END ===",
        msl.len(),
        msl
    );
}

#[test]
fn test_dump_full_msl_apple8() {
    let config = MetalGemmConfig::default_apple8_f16();
    let msl = build_standalone_gemm(&config);
    eprintln!(
        "\n=== Generated MSL (apple8 f16, {} bytes) ===\n{}\n=== END ===",
        msl.len(),
        msl
    );
}

#[test]
fn test_apple8_msl_compiles_with_xcrun_metal() {
    // Apple8 (polyfill) path uses no AIR intrinsics and can be validated
    // by the offline Metal compiler. Apple9 (hardware async) requires runtime
    // compilation via MTLDevice.makeLibrary() due to __asm declarations.
    let config = MetalGemmConfig::default_apple8_f16();
    let msl = build_standalone_gemm(&config);

    let tmp = std::env::temp_dir().join("ferrite_test_apple8.metal");
    std::fs::write(&tmp, &msl).expect("write temp MSL");

    let output = std::process::Command::new("xcrun")
        .args(["metal", "-std=metal3.0", "-Werror", "-c"])
        .arg(&tmp)
        .arg("-o")
        .arg("/dev/null")
        .output()
        .expect("xcrun metal must be available on macOS");

    std::fs::remove_file(&tmp).ok();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("Metal shader compilation failed:\n{}", stderr);
    }
}

#[test]
fn test_store_phase_uses_apply_offset() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);

    // Store phase should compute C_dst via apply_offset
    let store_section = msl.rfind("Store accumulators").unwrap();
    let store_body = &msl[store_section..];
    assert!(
        store_body.contains("apply_offset"),
        "Store must use apply_offset for C address"
    );
    assert!(
        store_body.contains("C_acc->store"),
        "Store must call store on accumulators"
    );
}
