use ferrite_metal::config::{GpuGeneration, MetalGemmConfig, Precision};
use ferrite_metal::emitter::build_standalone_gemm;

// ═══════════════════════════════════════════════════════════════════
// Config validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_apple8_tile_sizes() {
    let c = MetalGemmConfig::default_apple8_f16();
    assert_eq!(c.block_m, 48);
    assert_eq!(c.block_n, 48);
    assert_eq!(c.block_k, 32);
    assert!(!c.prefer_async_load);
    assert_eq!(c.gpu_gen, GpuGeneration::Apple8);
}

#[test]
fn test_apple9_tile_sizes() {
    let c = MetalGemmConfig::default_apple9_f16();
    assert_eq!(c.block_m, 32);
    assert_eq!(c.block_n, 32);
    assert_eq!(c.block_k, 8);
    assert!(c.prefer_async_load);
    assert_eq!(c.gpu_gen, GpuGeneration::Apple9);
}

#[test]
fn test_apple9_leading_block_dims() {
    let c = MetalGemmConfig::default_apple9_f16();
    assert_eq!(c.leading_block_dims, Some([32, 32, 32]));
    assert_eq!(c.leading_block_dim('A'), 32);
    assert_eq!(c.leading_block_dim('B'), 32);
    assert_eq!(c.leading_block_dim('C'), 32);
}

#[test]
fn test_register_dimensions() {
    let c = MetalGemmConfig::default_apple9_f16();
    assert_eq!(c.register_m(), 32);
    assert_eq!(c.register_n(), 32);

    let mut c2 = c.clone();
    c2.splits = [2, 2];
    assert_eq!(c2.register_m(), 16);
    assert_eq!(c2.register_n(), 16);
}

#[test]
fn test_block_bytes_f16() {
    let c = MetalGemmConfig::default_apple9_f16();
    // A: not transposed [M=32 rows × lead=32 cols], lead=32, trailing=M=32
    assert_eq!(c.block_bytes('A'), 2048, "A: 32×32×2");
    // B: transposed [K=8 rows × lead=32 cols], lead=32, trailing=K=8
    assert_eq!(c.block_bytes('B'), 512, "B: 32×8×2");
}

#[test]
fn test_block_bytes_f16_apple8() {
    let c = MetalGemmConfig::default_apple8_f16();
    // A: not transposed, leading = blockK = 32, trailing = blockM = 48
    assert_eq!(c.block_bytes('A'), 3072, "A: 32×48×2");
    // B: transposed, leading = blockN = 48, trailing = blockK = 32
    // (leading_block_dim('B') = block_n = 48 for transposed B)
    assert_eq!(c.block_bytes('B'), 3072, "B: 48×32×2");
}

#[test]
fn test_threadgroup_memory_apple9() {
    let c = MetalGemmConfig::default_apple9_f16();
    let mem = c.threadgroup_memory();
    // A=2048 + B=512 = 2560 (C stays in registers)
    assert_eq!(mem, 2560);
}

#[test]
fn test_threadgroup_memory_apple8() {
    let c = MetalGemmConfig::default_apple8_f16();
    let mem = c.threadgroup_memory();
    // A=3072 + B=3072 = 6144
    assert_eq!(mem, 6144);
}

#[test]
fn test_threadgroup_size() {
    let c = MetalGemmConfig::default_apple9_f16();
    assert_eq!(c.threadgroup_size(), 32);
    let mut c2 = c.clone();
    c2.splits = [2, 2];
    assert_eq!(c2.threadgroup_size(), 128);
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
    let mut c = MetalGemmConfig::default_apple8_f16();
    c.leading_block_dims = None;

    c.transpose[0] = false;
    assert_eq!(c.leading_block_dim('A'), c.block_k);
    c.transpose[0] = true;
    assert_eq!(c.leading_block_dim('A'), c.block_m);

    c.transpose[1] = false;
    assert_eq!(c.leading_block_dim('B'), c.block_k);
    c.transpose[1] = true;
    assert_eq!(c.leading_block_dim('B'), c.block_n);
}

// ═══════════════════════════════════════════════════════════════════
// MSL generation — structural correctness (native simdgroup_matrix API)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_generates_valid_msl_structure() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);

    assert!(
        msl.contains("#include <metal_stdlib>"),
        "Missing metal include"
    );
    assert!(msl.contains("using namespace metal;"), "Missing namespace");
    assert!(msl.contains("kernel void gemm("), "Missing kernel");
    assert!(msl.contains("[[buffer(0)]]"), "Missing buffer 0");
    assert!(msl.contains("[[buffer(1)]]"), "Missing buffer 1");
    assert!(msl.contains("[[buffer(2)]]"), "Missing buffer 2");

    // Native simdgroup_matrix API
    assert!(
        msl.contains("simdgroup_matrix<"),
        "Missing simdgroup_matrix type"
    );
    assert!(msl.contains("simdgroup_load"), "Missing simdgroup_load");
    assert!(msl.contains("simdgroup_store"), "Missing simdgroup_store");
    assert!(
        msl.contains("simdgroup_multiply_accumulate"),
        "Missing multiply_accumulate"
    );

    // K-loop and memory
    assert!(
        msl.contains("for (uint k = 0; k < K; k += K_group)"),
        "Missing K-loop"
    );
    assert!(msl.contains("threadgroup_block"), "Missing threadgroup");
    assert!(msl.contains("threadgroup_barrier"), "Missing barrier");
    assert!(msl.contains("C_sram"), "Missing accumulators");
}

#[test]
fn test_constants_match_config() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
    assert!(msl.contains("M_group = 32"));
    assert!(msl.contains("N_group = 32"));
    assert!(msl.contains("K_group = 8"));
}

#[test]
fn test_correct_precision_types_in_msl() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
    assert!(msl.contains("half"), "Should use half for f16");
    assert!(
        msl.contains("float"),
        "Should use float for f32 accumulator"
    );
}

#[test]
fn test_msl_output_not_empty() {
    for config in [
        MetalGemmConfig::default_apple8_f16(),
        MetalGemmConfig::default_apple9_f16(),
    ] {
        let msl = build_standalone_gemm(&config);
        assert!(msl.len() > 500, "Got {} bytes", msl.len());
    }
}

#[test]
fn test_msl_balanced_braces() {
    for config in [
        MetalGemmConfig::default_apple8_f16(),
        MetalGemmConfig::default_apple9_f16(),
    ] {
        let msl = build_standalone_gemm(&config);
        let opens = msl.chars().filter(|c| *c == '{').count();
        let closes = msl.chars().filter(|c| *c == '}').count();
        assert_eq!(opens, closes, "Unbalanced: {} vs {}", opens, closes);
    }
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
            "Unreplaced template: {}",
            msl.lines().find(|l| l.contains("{{")).unwrap_or("???")
        );
    }
}

#[test]
fn test_pipeline_stage_ordering() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);

    let const_pos = msl.find("M_group").unwrap();
    let kernel_pos = msl.find("kernel void gemm").unwrap();
    let setup_pos = msl.find("M_offset").unwrap();
    let acc_pos = msl.find("C_sram").unwrap();
    let kloop_pos = msl.find("for (uint k = 0").unwrap();
    let store_pos = msl.find("simdgroup_store").unwrap();

    assert!(const_pos < kernel_pos, "Constants before kernel");
    assert!(kernel_pos < setup_pos, "Kernel before setup");
    assert!(setup_pos < acc_pos, "Setup before accumulators");
    assert!(acc_pos < kloop_pos, "Accumulators before K-loop");
    assert!(kloop_pos < store_pos, "K-loop before store");
}

#[test]
fn test_k_loop_has_correct_structure() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);

    let kloop_start = msl.find("for (uint k = 0; k < K; k += K_group)").unwrap();
    let kloop_body = &msl[kloop_start..];

    assert!(kloop_body.contains("A_block"), "Must have A_block");
    assert!(kloop_body.contains("B_block"), "Must have B_block");
    assert!(
        kloop_body.contains("threadgroup_barrier"),
        "Must have barrier"
    );
    assert!(
        kloop_body.contains("simdgroup_load"),
        "Must have simdgroup_load"
    );
    assert!(
        kloop_body.contains("simdgroup_multiply_accumulate"),
        "Must have MMA"
    );
}

#[test]
fn test_dump_full_msl_apple9() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
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
