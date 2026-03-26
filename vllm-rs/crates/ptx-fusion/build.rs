// Build script: compile vllm-cuda kernels to PTX for fusion testing.
// Only runs when the "cuda" feature is enabled.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Only compile PTX when cuda feature is active
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let vllm_cuda_csrc = manifest_dir
        .parent()
        .unwrap()
        .join("vllm-cuda")
        .join("csrc");
    let out_dir = manifest_dir.join("kernels");

    // Find nvcc
    let nvcc = find_nvcc();

    // Compile rms_norm_kernel<float>
    compile_ptx(
        &nvcc,
        &vllm_cuda_csrc,
        &out_dir,
        "vllm_rms_norm",
        r#"
#include "layernorm_kernels.cu"
template __global__ void rms_norm_kernel<float>(
    float* __restrict__, const float* __restrict__,
    const float* __restrict__, float, int);
"#,
    );

    // Compile act_and_mul_kernel<silu, float>
    compile_ptx(
        &nvcc,
        &vllm_cuda_csrc,
        &out_dir,
        "vllm_silu_mul",
        r#"
#include "activation_kernels.cu"
template __global__ void act_and_mul_kernel<silu, float>(
    float* __restrict__, const float* __restrict__,
    const float* __restrict__, int);
"#,
    );
}

fn find_nvcc() -> String {
    // Try CUDA_HOME first, then common paths
    if let Ok(home) = std::env::var("CUDA_HOME") {
        let p = format!("{home}/bin/nvcc");
        if std::path::Path::new(&p).exists() {
            return p;
        }
    }
    for ver in ["12.9", "12.8", "12.6", "12.4", "12.2", "12.0"] {
        let p = format!("/usr/local/cuda-{ver}/bin/nvcc");
        if std::path::Path::new(&p).exists() {
            return p;
        }
    }
    // Fallback to PATH
    "nvcc".to_string()
}

fn compile_ptx(
    nvcc: &str,
    include_dir: &std::path::Path,
    out_dir: &std::path::Path,
    name: &str,
    source: &str,
) {
    let tmp_cu = std::env::temp_dir().join(format!("{name}_ferrite.cu"));
    let out_ptx = out_dir.join(format!("{name}.ptx"));

    std::fs::write(&tmp_cu, source).expect("write temp .cu");

    println!("cargo:rerun-if-changed={}", include_dir.display());

    let output = Command::new(nvcc)
        .args([
            "-ptx",
            "-arch=sm_89",
            &format!("-I{}", include_dir.display()),
            tmp_cu.to_str().unwrap(),
            "-o",
            out_ptx.to_str().unwrap(),
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            println!(
                "cargo:warning=compiled {name}.ptx ({} bytes)",
                std::fs::metadata(&out_ptx).map(|m| m.len()).unwrap_or(0)
            );
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            println!("cargo:warning=nvcc failed for {name}: {stderr}");
            // Don't panic — tests that need these PTX files will skip
        }
        Err(e) => {
            println!("cargo:warning=nvcc not found for {name}: {e}");
        }
    }

    let _ = std::fs::remove_file(&tmp_cu);
}
