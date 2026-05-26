// SPDX-License-Identifier: Apache-2.0
// Link directives for the ferrite-mega-cu-emit binary.
//
// ferrite-cuda-builder's build.rs compiles the .cu sources into
// static .a files in the cudaforge cache (`~/.cache/cudaforge/vllm-cuda/`).
// This build.rs emits the rustc link flags that pull those libs in so
// extern "C" symbols ferrite-kernels references resolve at link time.
//
// Mirrors `crates/ferrite-cost-sweep/build.rs`. The bin itself does
// no CUDA work — it just walks the `MegaCanonicalEmit` inventory and
// writes `.cu` files — but ferrite-models' lib transitively pulls in
// ferrite-kernels' externs, so the linker still has to resolve them.

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

    // Same lib set ferrite-cost-sweep links — covers every extern
    // ferrite-kernels declares (vllm fused kernels, GGML, marlin,
    // CUTLASS GEMM family, flash-attn, flashinfer).
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
    // Link the previously-compiled megakernels archive so the
    // `forward_mega_<canonical>()` functions ferrite-models emits
    // (which extern-call `<canonical>_launch_host`) resolve at link
    // time. The .a we link here is the OLD one (from a prior build);
    // running this binary writes fresh .cu, and a subsequent
    // ferrite-cuda-builder pass rebuilds the .a from those.
    println!("cargo:rustc-link-lib=static=megakernels");

    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".into());
    println!("cargo:rustc-link-search={cuda_path}/lib64");
    println!("cargo:rustc-link-search={cuda_path}/lib");
}
