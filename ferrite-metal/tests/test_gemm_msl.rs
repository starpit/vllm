use ferrite_metal::config::{MetalGemmConfig, Precision, GpuGeneration};
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
    assert_eq!(b_bytes, 2048, "B block bytes for transposed 32×32 f16 with leading=32");
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
    assert_eq!(b_bytes, 3072, "B block bytes for 48×32 f16 transposed auto-leading");
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
    assert!(msl.contains("#include <metal_stdlib>"), "Missing metal include");
    assert!(msl.contains("using namespace metal;"), "Missing namespace");

    // Must have kernel declaration
    assert!(msl.contains("kernel void gemm("), "Missing kernel declaration");

    // Must have buffer bindings
    assert!(msl.contains("[[buffer(0)]]"), "Missing buffer binding 0");
    assert!(msl.contains("[[buffer(1)]]"), "Missing buffer binding 1");
    assert!(msl.contains("[[buffer(2)]]"), "Missing buffer binding 2");

    // Must have simdgroup intrinsics
    assert!(msl.contains("simdgroup_matrix_storage"), "Missing simdgroup_matrix_storage");
    assert!(msl.contains("morton_order"), "Missing morton_order helper");
    assert!(msl.contains("get_sram"), "Missing get_sram helper");

    // Must have K-loop
    assert!(msl.contains("for (uint k = 0; k < K; k += K_group)"), "Missing K-loop");

    // Must have threadgroup memory
    assert!(msl.contains("threadgroup_block"), "Missing threadgroup allocation");
    assert!(msl.contains("threadgroup_barrier"), "Missing barrier");

    // Must have accumulator init and store
    assert!(msl.contains("C_sram"), "Missing C_sram accumulators");
    assert!(msl.contains("store"), "Missing store");
}

#[test]
fn test_apple9_uses_async_copy() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
    assert!(msl.contains("simdgroup_event"), "apple9 should use async copy");
    assert!(msl.contains("async_copy"), "apple9 should use async_copy");
}

#[test]
fn test_apple8_uses_direct_load() {
    let config = MetalGemmConfig::default_apple8_f16();
    let msl = build_standalone_gemm(&config);
    // apple8 path: no async copy, direct load from device memory
    assert!(!msl.contains("simdgroup_event"), "apple8 should NOT use async copy");
}

#[test]
fn test_correct_precision_types_in_msl() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
    assert!(msl.contains("half"), "Should use half for f16 operands");
    assert!(msl.contains("float"), "Should use float for f32 accumulator");
}

#[test]
fn test_constants_match_config() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
    assert!(msl.contains("M_group = 32"), "M_group should be 32 for apple9");
    assert!(msl.contains("N_group = 32"), "N_group should be 32 for apple9");
    assert!(msl.contains("K_group = 8"), "K_group should be 8 for apple9");
}

#[test]
fn test_msl_output_not_empty() {
    for config in [
        MetalGemmConfig::default_apple8_f16(),
        MetalGemmConfig::default_apple9_f16(),
    ] {
        let msl = build_standalone_gemm(&config);
        assert!(msl.len() > 500, "Generated MSL should be substantial, got {} bytes", msl.len());
    }
}

#[test]
fn test_msl_balanced_braces() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
    let opens = msl.chars().filter(|c| *c == '{').count();
    let closes = msl.chars().filter(|c| *c == '}').count();
    assert_eq!(opens, closes, "Unbalanced braces: {} opens vs {} closes", opens, closes);
}

#[test]
fn test_no_template_variables_remain() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
    assert!(!msl.contains("{{"), "Unreplaced template variable found in output: {}",
        msl.lines().find(|l| l.contains("{{")).unwrap_or("???"));
}
