// SPDX-License-Identifier: Apache-2.0
// Link directives for the cost-sweep binary.
//
// ferrite-cuda-builder's build.rs compiles the .cu sources into
// static .a files in the cudaforge cache (`~/.cache/cudaforge/vllm-cuda/`).
// This build.rs emits the rustc link flags that pull those libs in so
// extern "C" symbols (cutlass_gemm_*_launch, rms_norm_bf16, etc.)
// resolve at link time.
//
// Mirrors the core set of libs that `vllm-cuda/build.rs` emits —
// trimmed to only the libs cost-sweep actually needs (no marlin,
// no scaled_mm). The GEMM sweep links `vllm_kernels` +
// `cutlass_standalone_gemm`; the attention sweep additionally links
// `vllm_flash_attn` (FA2 baseline) and `flashinfer_attn` (per-tuple
// FI shim built by ferrite-cuda-builder).

fn main() {
    #[cfg(feature = "cuda")]
    cuda_link();

    #[cfg(not(feature = "cuda"))]
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn cuda_link() {
    let cache_dir = dirs::cache_dir()
        .expect("no cache directory found")
        .join("cudaforge")
        .join("vllm-cuda");
    let cache_str = cache_dir.to_string_lossy().to_string();

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-search={}", cache_str);

    // Static kernel libs consumed by the GEMM sweep. These are all
    // produced by ferrite-cuda-builder's build.rs; the dependency
    // edge in Cargo.toml forces that crate's build to run first.
    println!("cargo:rustc-link-lib=static=vllm_kernels");
    println!("cargo:rustc-link-lib=static=ggml_kernels");
    println!("cargo:rustc-link-lib=static=marlin_kernels");
    println!("cargo:rustc-link-lib=static=marlin_moe_kernels");
    println!("cargo:rustc-link-lib=static=cutlass_scaled_mm");
    println!("cargo:rustc-link-lib=static=cutlass_standalone_gemm");
    println!("cargo:rustc-link-lib=static=cutlass_gemm_silu_mul");
    println!("cargo:rustc-link-lib=static=cutlass_gemm_bias");
    println!("cargo:rustc-link-lib=static=vllm_flash_attn");
    println!("cargo:rustc-link-lib=static=flashinfer_attn");

    // CUDA runtime. cudart_static requires rt + dl; cublas stays
    // dynamic (and is needed for the cuBLAS baseline row).
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Toolkit lib path — follow the same env-var fallback chain as
    // vllm-cuda's link setup.
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".into());
    println!("cargo:rustc-link-search={cuda_path}/lib64");
    println!("cargo:rustc-link-search={cuda_path}/lib");
}
