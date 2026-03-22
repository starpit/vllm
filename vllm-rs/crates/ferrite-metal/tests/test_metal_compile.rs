/// Runtime Metal compilation tests.
///
/// These tests use MTLDevice.newLibraryWithSource() to compile generated MSL
/// on actual Apple GPU hardware and verify the kernel can be loaded into a
/// compute pipeline.
///
/// The apple9 hardware async copy path uses __asm AIR intrinsic declarations
/// which are broken on Metal 4 / macOS 26 (Apple changed the compiler).
/// On such systems, we fall back to the polyfill path which uses loop-based
/// copies instead of hardware DMA. The tile sizes remain the same.
use ferrite_metal::config::MetalGemmConfig;
use ferrite_metal::gemm::build_standalone_gemm;
use metal::{CompileOptions, Device, MTLLanguageVersion};

fn get_device() -> metal::Device {
    Device::system_default().expect("No Metal device — cannot run GPU tests")
}

fn try_compile(device: &metal::Device, source: &str) -> Result<metal::Library, String> {
    let options = CompileOptions::new();
    options.set_language_version(MTLLanguageVersion::V3_0);
    device.new_library_with_source(source, &options)
}

fn compile_and_get_pipeline(device: &metal::Device, source: &str) -> metal::ComputePipelineState {
    let library = try_compile(device, source).expect("MSL compilation failed");
    let func = library
        .get_function("gemm", None)
        .expect("'gemm' kernel function not found in compiled library");
    device
        .new_compute_pipeline_state_with_function(&func)
        .expect("Failed to create compute pipeline from 'gemm' function")
}

/// Returns true if the Metal compiler on this system supports MFA's
/// __asm("air.simdgroup_async_copy...") intrinsic declarations.
/// Metal 4 / macOS 26 broke this syntax.
fn supports_asm_air_intrinsics(device: &metal::Device) -> bool {
    let test_source = r#"
struct _simdgroup_event_t;
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_1d(
  ulong, ulong, threadgroup void *, const device void *, ulong)
  __asm("air.simdgroup_async_copy_1d.p3i8.p1i8");
kernel void test() {}
"#;
    let options = CompileOptions::new();
    options.set_language_version(MTLLanguageVersion::V3_0);
    device
        .new_library_with_source(test_source, &options)
        .is_ok()
}

// ═══════════════════════════════════════════════════════════════════
// Apple8 config (polyfill path) — always works
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_apple8_f16_compiles_on_gpu() {
    let device = get_device();
    let config = MetalGemmConfig::default_apple8_f16();
    let msl = build_standalone_gemm(&config);
    try_compile(&device, &msl).expect("Apple8 f16 MSL must compile on GPU");
}

#[test]
fn test_apple8_f16_creates_pipeline() {
    let device = get_device();
    let config = MetalGemmConfig::default_apple8_f16();
    let msl = build_standalone_gemm(&config);
    let _pipeline = compile_and_get_pipeline(&device, &msl);
}

// ═══════════════════════════════════════════════════════════════════
// Apple9 config with polyfill — uses apple9 tile sizes but polyfill copy
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_apple9_polyfill_compiles_on_gpu() {
    let device = get_device();
    let mut config = MetalGemmConfig::default_apple9_f16();
    config.prefer_async_load = false; // Force polyfill path
    let msl = build_standalone_gemm(&config);
    try_compile(&device, &msl).expect("Apple9 polyfill MSL must compile on GPU");
}

#[test]
fn test_apple9_polyfill_creates_pipeline() {
    let device = get_device();
    let mut config = MetalGemmConfig::default_apple9_f16();
    config.prefer_async_load = false;
    let msl = build_standalone_gemm(&config);
    let _pipeline = compile_and_get_pipeline(&device, &msl);
}

// ═══════════════════════════════════════════════════════════════════
// Apple9 config with hardware async — only on systems with __asm support
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_apple9_hardware_async_compiles_if_supported() {
    let device = get_device();

    if !supports_asm_air_intrinsics(&device) {
        eprintln!(
            "SKIPPED: Metal compiler on this system does not support __asm AIR intrinsics \
             (Metal 4 / macOS 26 broke this). Using polyfill path instead."
        );
        // Verify the polyfill fallback works with apple9 tile sizes
        let mut config = MetalGemmConfig::default_apple9_f16();
        config.prefer_async_load = false;
        let msl = build_standalone_gemm(&config);
        try_compile(&device, &msl).expect("Apple9 polyfill fallback must compile");
        return;
    }

    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_standalone_gemm(&config);
    try_compile(&device, &msl).expect("Apple9 hardware async MSL must compile on GPU");
}

// ═══════════════════════════════════════════════════════════════════
// Offline compiler validation (xcrun metal)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_apple8_compiles_with_xcrun_metal() {
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
        panic!("Metal offline compilation failed:\n{}", stderr);
    }
}

#[test]
fn test_apple9_polyfill_compiles_with_xcrun_metal() {
    let mut config = MetalGemmConfig::default_apple9_f16();
    config.prefer_async_load = false;
    let msl = build_standalone_gemm(&config);

    let tmp = std::env::temp_dir().join("ferrite_test_apple9_polyfill.metal");
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
        panic!("Metal offline compilation failed:\n{}", stderr);
    }
}
