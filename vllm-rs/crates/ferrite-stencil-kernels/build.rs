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

    // Stencil smoke kernel .a is sm_89-only (the legacy 3b smoke
    // kernel; emit_kittens is the sm_90a+ path). Link conditionally
    // so the crate still builds on H100 where ferrite-cuda-builder
    // early-returned without producing the .a.
    let stencil_lib = std::path::Path::new(&cache_str).join("libstencil_kernels.a");
    if stencil_lib.exists() {
        println!("cargo:rustc-link-lib=static=stencil_kernels");
        println!("cargo:rustc-cfg=stencil_linked");
    }
    println!("cargo:rustc-check-cfg=cfg(stencil_linked)");

    // Kittens megakernel .a only exists on sm_90a+ builds where
    // ferrite-cuda-builder compiled it. When present, link it and
    // set the `kittens_linked` cfg so tests can gate the FFI calls
    // on it; when absent, skip both so the crate still builds on
    // sm_89 dev boxes.
    let kittens_lib = std::path::Path::new(&cache_str).join("libkittens_kernels.a");
    println!(
        "cargo:warning=ferrite-stencil-kernels: checking {kittens_lib:?} exists={}",
        kittens_lib.exists(),
    );
    if kittens_lib.exists() {
        println!("cargo:rustc-link-lib=static=kittens_kernels");
        println!("cargo:rustc-cfg=kittens_linked");
        println!("cargo:warning=ferrite-stencil-kernels: kittens_linked cfg set");
    }
    // Declare the cfg to rustc so `cfg(kittens_linked)` doesn't
    // trigger "unexpected_cfgs" on newer rustc.
    println!("cargo:rustc-check-cfg=cfg(kittens_linked)");

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
