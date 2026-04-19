// SPDX-License-Identifier: Apache-2.0
// Link directives for ferrite-stencil-kernels.
//
// ferrite-cuda-builder compiled the stencil .cu sources into
// `libstencil_kernels.a` in the cudaforge cache. This build.rs
// wires the link flags so the extern "C" symbols in src/lib.rs
// resolve when the test binary links.
//
// Modelled on ferrite-cost-sweep/build.rs — same cache path,
// same cudart-static + rt/dl/stdc++ set.

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

    println!("cargo:rustc-link-lib=static=stencil_kernels");

    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".into());
    println!("cargo:rustc-link-search={cuda_path}/lib64");
    println!("cargo:rustc-link-search={cuda_path}/lib");
}
