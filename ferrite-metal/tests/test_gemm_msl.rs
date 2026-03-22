use ferrite_metal::config::MetalGemmConfig;
use ferrite_metal::gemm::build_standalone_gemm;

#[test]
fn test_generates_msl() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);

    // Should contain kernel declaration
    assert!(msl.contains("kernel void gemm("), "Missing kernel declaration");

    // Should contain simdgroup matrix types
    assert!(msl.contains("simdgroup_matrix_storage"), "Missing simdgroup types");

    // Should contain K-loop
    assert!(msl.contains("for (uint k = 0; k < K; k += K_group)"), "Missing K-loop");

    // Should contain multiply
    assert!(msl.contains("multiply"), "Missing multiply call");

    // Should contain threadgroup memory
    assert!(msl.contains("threadgroup_block"), "Missing threadgroup memory");

    // Print for inspection
    println!("Generated MSL ({} bytes):\n{}", msl.len(), msl);
}

#[test]
fn test_apple8_config() {
    let config = MetalGemmConfig::default_apple8_f16();
    assert_eq!(config.block_m, 48);
    assert_eq!(config.block_n, 48);
    assert_eq!(config.block_k, 32);
    assert!(!config.prefer_async_load);
}

#[test]
fn test_apple9_config() {
    let config = MetalGemmConfig::default_apple9_f16();
    assert_eq!(config.block_m, 32);
    assert_eq!(config.block_n, 32);
    assert_eq!(config.block_k, 8);
    assert!(config.prefer_async_load);
}

#[test]
fn test_threadgroup_memory() {
    let config = MetalGemmConfig::default_apple8_f16();
    let mem = config.threadgroup_memory();
    // 48×32×2 (A) + 48×32×2 (B) = 6144 bytes for f16 48×48×32
    assert!(mem > 0);
    println!("Threadgroup memory: {} bytes", mem);
}
