// SPDX-License-Identifier: Apache-2.0
// Links the CUDA kernel .a files compiled by ferrite-cuda-builder's build.rs.
// Kernel compilation lives in that crate; this build.rs only emits linker flags.

fn main() {
    #[cfg(feature = "cuda")]
    cuda_link();

    #[cfg(not(feature = "cuda"))]
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn cuda_link() {
    // Locate the shared cudaforge cache populated by ferrite-cuda-builder's build.rs.
    let cache_dir = dirs::cache_dir()
        .expect("no cache directory found")
        .join("cudaforge")
        .join("vllm-cuda");
    let cache_str = cache_dir.to_string_lossy().to_string();

    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rustc-link-search={}", cache_str);

    // Historical megakernel .a (retired with the pre-Ferrite TK
    // runtime). Kept as a conditional link so rebuilding against a
    // cache that still contains `libmegakernels.a` doesn't fail the
    // linker; present builds skip the library entirely.
    let mk_lib = std::path::Path::new(&cache_str).join("libmegakernels.a");
    if mk_lib.exists() {
        println!("cargo:rustc-link-lib=static=megakernels");
    }

    // Kittens-based megakernel .a (sm_90a+). Only present when the
    // builder ran on H100+ hardware with THUNDERKITTENS_ROOT set.
    // Skip link on lower arches so the build still succeeds there.
    let kittens_lib = std::path::Path::new(&cache_str).join("libkittens_kernels.a");
    if kittens_lib.exists() {
        println!("cargo:rustc-link-lib=static=kittens_kernels");
    }

    println!("cargo:rustc-link-lib=static=vllm_kernels");
    println!("cargo:rustc-link-lib=static=ggml_kernels");
    println!("cargo:rustc-link-lib=static=marlin_kernels");
    println!("cargo:rustc-link-lib=static=marlin_moe_kernels");
    println!("cargo:rustc-link-lib=static=cutlass_scaled_mm");
    println!("cargo:rustc-link-lib=static=cutlass_standalone_gemm");
    println!("cargo:rustc-link-lib=static=vllm_flash_attn");
    println!("cargo:rustc-link-lib=static=flashinfer_attn");

    // cudart_static requires rt + dl; cublas/cublasLt remain dynamic.
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
