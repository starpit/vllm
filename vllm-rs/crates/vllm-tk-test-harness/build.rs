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

    let cu_files: Vec<String> = vec![
        shim_cu.display().to_string(),
        cutlass_gemm_cu.display().to_string(),
    ];

    // ── Include paths ──
    //
    // CUTLASS headers (used by both source files). Optional include path —
    // not set on all dev machines; if missing, cudaforge will surface the
    // error from nvcc.
    let cutlass_root = std::path::PathBuf::from(
        std::env::var("CUTLASS_ROOT").unwrap_or_else(|_| "/home/moosevan/cutlass".to_string()),
    );
    let cutlass_include = cutlass_root.join("include");
    let cutlass_tools_util = cutlass_root.join("tools/util/include");

    // FlashInfer headers — pinned via cudaforge git dependency.
    // Cudaforge clones+caches the repo at the pinned commit into
    // `~/.cudaforge/git/checkouts/flashinfer-<hash>/`, shared across
    // worktrees and version-locked by the SHA below. Used by the
    // attention shim.
    //
    // Bump this commit deliberately and rerun goldens.
    const FLASHINFER_COMMIT: &str = "08ab45d67705b301ee66e63c6999c934c72dd41c";

    let mut builder = cudaforge::KernelBuilder::new();
    builder = builder
        .out_dir(&cache_dir)
        .source_files(cu_files)
        .include_path(harness_csrc.display().to_string())
        .with_git_dependency(
            "flashinfer",
            "https://github.com/flashinfer-ai/flashinfer.git",
            FLASHINFER_COMMIT,
            vec!["include"],
            /*recurse_submodules=*/ false,
        );
    if cutlass_include.exists() {
        println!("cargo:warning=cutlass found at {}", cutlass_root.display());
        builder = builder
            .include_path(cutlass_include.display().to_string())
            .include_path(cutlass_tools_util.display().to_string());
    } else {
        println!(
            "cargo:warning=cutlass not found at {} (set CUTLASS_ROOT)",
            cutlass_root.display()
        );
    }
    builder
        .arg("-std=c++20")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-extended-lambda")
        .arg("--expt-relaxed-constexpr")
        .arg("-DNDEBUG")
        .arg("-Xcompiler=-fPIC")
        .arg("-Xcompiler=-fno-strict-aliasing")
        .arg("-Xcompiler=-Wno-psabi")
        .arg("-arch=sm_89")
        .arg("-lineinfo")
        .build_lib(format!("{cache_str}/libtk_test_ops.a"))
        .expect("failed to build solver-adjacent test kernels");

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

    // Rerun triggers
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", shim_cu.display());
    println!("cargo:rerun-if-changed={}", cutlass_gemm_cu.display());
}
