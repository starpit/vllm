// SPDX-License-Identifier: Apache-2.0
//! Build script for the solver-adjacent GPU test harness.
//!
//! With `--features cuda`:
//!   1. Compiles the canonical CUTLASS standalone GEMM TU
//!      (`vllm-cuda/csrc/cutlass_standalone_gemm.cu`) — provides the
//!      `cutlass_gemm_*_launch` and `cutlass_gemv_launch` symbols that
//!      `tests/gpu_cost_sweep.rs` benchmarks.
//!   2. Compiles the FlashInfer attention shim
//!      (`csrc/flashinfer_attention_shim.cu`) — provides
//!      `run_flashinfer_attention_smoke` for
//!      `tests/flashinfer_attention_test.rs`.
//!   3. Links against cuBLAS (for the cuBLAS baseline in the cost sweep)
//!      and vllm-rs's `libvllm_kernels.a` (transitively needed by the
//!      FlashInfer shim, depending on the flashinfer build flags).

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();
}

#[cfg(feature = "cuda")]
fn build_cuda() {
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

    // Set up cudaforge cache directory
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("cudaforge")
        .join("vllm-tk-test-harness");
    std::fs::create_dir_all(&cache_dir).ok();
    let cache_str = cache_dir.display().to_string();

    // ── Source files ──
    let harness_csrc = manifest_dir.join("csrc");
    let shim_cu = harness_csrc.join("flashinfer_attention_shim.cu");
    let cutlass_gemm_cu = workspace_root.join("crates/vllm-cuda/csrc/cutlass_standalone_gemm.cu");
    let cutlass_silu_mul_cu = workspace_root.join("crates/vllm-cuda/csrc/cutlass_gemm_silu_mul.cu");

    let barrier_cu = harness_csrc.join("barrier_sweep.cu");
    let cu_files: Vec<String> = vec![
        shim_cu.display().to_string(),
        cutlass_gemm_cu.display().to_string(),
        cutlass_silu_mul_cu.display().to_string(),
        barrier_cu.display().to_string(),
    ];

    // ── CUTLASS via cudaforge ──
    // Same commit as vllm-kernels-cuda uses for the standalone GEMM.
    // CUTLASS 4.2.1 — has full sm90 (Hopper) support.
    const CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

    // FlashInfer headers — pinned via cudaforge git dependency.
    const FLASHINFER_COMMIT: &str = "08ab45d67705b301ee66e63c6999c934c72dd41c";

    let arch = detect_cuda_arch();
    let arch_num: u32 = arch.parse().unwrap_or(89);

    // sm90+ needs c++20 for CuTe; sm89 and below use c++17.
    let std_flag = if arch_num >= 90 {
        "-std=c++20"
    } else {
        "-std=c++17"
    };

    // ── Build 0 (optional): megakernel .cu files from proc-macro cache ──
    let megakernel_cache = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("cudaforge/megakernels");
    let megakernel_cus: Vec<String> = if megakernel_cache.exists() {
        std::fs::read_dir(&megakernel_cache)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "cu"))
            .map(|e| e.path().display().to_string())
            .collect()
    } else {
        vec![]
    };
    if !megakernel_cus.is_empty() {
        let vllm_cuda_csrc = workspace_root.join("crates/vllm-cuda/csrc");
        let mut mk_builder = cudaforge::KernelBuilder::new();
        mk_builder = mk_builder
            .out_dir(&cache_dir)
            .source_files(megakernel_cus.clone())
            .include_path(vllm_cuda_csrc.display().to_string())
            .with_cutlass(Some(CUTLASS_COMMIT));
        mk_builder
            .arg(std_flag)
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("--expt-extended-lambda")
            .arg("--expt-relaxed-constexpr")
            .arg("-DNDEBUG")
            .arg("-Xcompiler=-fPIC")
            .arg("-Xcompiler=-fno-strict-aliasing")
            .arg("-Xcompiler=-Wno-psabi")
            .arg(&format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
            .arg("-lineinfo")
            .build_lib(format!("{cache_str}/libmegakernels.a"))
            .expect("failed to build megakernel .cu files");
        println!("cargo:rustc-link-lib=static=megakernels");
        for cu in &megakernel_cus {
            println!("cargo:rerun-if-changed={cu}");
        }
    }

    // ── Build 1: CUTLASS + FlashInfer (existing kernels) ──
    let mut builder = cudaforge::KernelBuilder::new();
    builder = builder
        .out_dir(&cache_dir)
        .source_files(cu_files)
        .include_path(harness_csrc.display().to_string())
        .with_cutlass(Some(CUTLASS_COMMIT))
        .with_git_dependency(
            "flashinfer",
            "https://github.com/flashinfer-ai/flashinfer.git",
            FLASHINFER_COMMIT,
            vec!["include"],
            /*recurse_submodules=*/ false,
        );

    builder
        .arg(std_flag)
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-extended-lambda")
        .arg("--expt-relaxed-constexpr")
        .arg("-DNDEBUG")
        .arg("-Xcompiler=-fPIC")
        .arg("-Xcompiler=-fno-strict-aliasing")
        .arg("-Xcompiler=-Wno-psabi")
        .arg(&format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
        .arg("-lineinfo")
        .build_lib(format!("{cache_str}/libtk_test_ops.a"))
        .expect("failed to build solver-adjacent test kernels");

    // ── Build 2: ThunderKittens GEMM (sm90+ only) ──
    if arch_num >= 90 {
        build_tk_gemm(&cache_dir, &cache_str, &harness_csrc, &arch);
    }

    // ── Link directives ──
    println!("cargo:rustc-link-search={cache_str}");
    println!("cargo:rustc-link-lib=static=tk_test_ops");

    // vllm-rs's fused kernel static lib (built by the sibling
    // vllm-kernels-cuda crate). Linked unconditionally because the
    // FlashInfer shim TU can reach vllm_kernels symbols via
    // transitive includes on some configurations.
    let vllm_kernels_dir = std::env::var("HOME")
        .map(|h| format!("{h}/.cache/cudaforge/vllm-cuda"))
        .unwrap_or_else(|_| "/tmp/cudaforge/vllm-cuda".into());
    println!("cargo:rustc-link-search={vllm_kernels_dir}");
    println!("cargo:rustc-link-lib=static=vllm_kernels");

    // CUDA toolkit
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".into());
    println!("cargo:rustc-link-search={cuda_path}/lib64");
    println!("cargo:rustc-link-search={cuda_path}/lib");
    println!("cargo:rustc-link-lib=static=cudart_static");
    // cuBLAS for the GEMM baseline in gpu_cost_sweep.
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=cuda");

    // Expose detected arch as cfg flag for conditional FFI/test code.
    println!("cargo:rustc-check-cfg=cfg(cuda_arch_sm90)");
    if arch_num >= 90 {
        println!("cargo:rustc-cfg=cuda_arch_sm90");
    }

    // Rerun triggers
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", shim_cu.display());
    println!("cargo:rerun-if-changed={}", cutlass_gemm_cu.display());
    let tk_cu = harness_csrc.join("tk_gemm_wrapper.cu");
    println!("cargo:rerun-if-changed={}", tk_cu.display());
    println!("cargo:rerun-if-changed={}", barrier_cu.display());
}

/// Build ThunderKittens GEMM wrapper (sm90+ only).
///
/// TK requires `-arch=sm_90a` (the `a` enables wgmma + TMA PTX),
/// which is incompatible with the CUTLASS TU's `-arch=sm_90`.
/// So we build it as a separate static lib.
#[cfg(feature = "cuda")]
fn build_tk_gemm(
    cache_dir: &std::path::Path,
    cache_str: &str,
    harness_csrc: &std::path::Path,
    _arch: &str,
) {
    // ThunderKittens source — expected at ~/git/ThunderKittens or
    // via TK_PATH env var.
    let tk_path = std::env::var("TK_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        format!("{home}/git/ThunderKittens")
    });
    let tk_root = std::path::PathBuf::from(&tk_path);
    if !tk_root.join("include/kittens.cuh").exists() {
        println!("cargo:warning=ThunderKittens not found at {tk_path}, skipping TK GEMM build");
        return;
    }
    println!("cargo:warning=Building ThunderKittens GEMM from {tk_path}");

    let tk_cu = harness_csrc.join("tk_gemm_wrapper.cu");
    let mut builder = cudaforge::KernelBuilder::new();
    builder = builder
        .out_dir(cache_dir)
        .source_files(vec![tk_cu.display().to_string()])
        .include_path(tk_root.join("include").display().to_string())
        .include_path(tk_root.join("prototype").display().to_string());

    builder
        .arg("-std=c++20")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-extended-lambda")
        .arg("--expt-relaxed-constexpr")
        .arg("-DNDEBUG")
        .arg("-DKITTENS_HOPPER")
        .arg("-Xcompiler=-fPIC")
        .arg("-Xcompiler=-fno-strict-aliasing")
        .arg("-Xcompiler=-Wno-psabi")
        .arg("-gencode=arch=compute_90a,code=sm_90a")
        .arg("-lineinfo")
        .build_lib(format!("{cache_str}/libtk_gemm.a"))
        .expect("failed to build ThunderKittens GEMM wrapper");

    println!("cargo:rustc-link-lib=static=tk_gemm");
}

/// Detect the GPU compute capability via nvidia-smi. Falls back to 89 (sm_89).
#[cfg(feature = "cuda")]
fn detect_cuda_arch() -> String {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output();
    if let Ok(out) = output {
        let s = String::from_utf8_lossy(&out.stdout);
        if let Some(line) = s.lines().next() {
            // "9.0" -> "90", "8.9" -> "89"
            let cleaned = line.trim().replace('.', "");
            if !cleaned.is_empty() {
                println!("cargo:warning=detected GPU compute capability: sm_{cleaned}");
                return cleaned;
            }
        }
    }
    println!("cargo:warning=nvidia-smi not found, defaulting to sm_89");
    "89".to_string()
}
