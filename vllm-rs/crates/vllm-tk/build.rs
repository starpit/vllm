// SPDX-License-Identifier: Apache-2.0
// Compiles the ThunderKittens KVM LLaMA sm89 megakernel into a static lib.
// Uses cudaforge for incremental builds (content-hashed, only recompiles on change).

fn main() {
    #[cfg(feature = "cuda")]
    cuda_build();

    #[cfg(not(feature = "cuda"))]
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn cuda_build() {
    // Use a stable shared cache so clippy/test/build share compiled .o files.
    let cache_dir = dirs::cache_dir()
        .expect("no cache directory found")
        .join("cudaforge")
        .join("vllm-tk");
    std::fs::create_dir_all(&cache_dir).expect("Failed to create cudaforge cache dir");
    let cache_str = cache_dir.to_string_lossy().to_string();

    let tk_include = "csrc/include";
    let tk_prototype = "csrc/prototype";

    // Track source files for cargo:rerun-if-changed.
    let source = "csrc/tk_launch.cu";
    let watch_files = [
        source,
        "csrc/llama_sm89.cuh",
        "csrc/rms_norm_sm89.cu",
        "csrc/qkv_rope_append_sm89.cu",
        "csrc/attention_prefill_sm89.cu",
        "csrc/attention_decode_sm89.cu",
        "csrc/matmul_adds_sm89.cu",
        "csrc/gate_silu_sm89.cu",
        "csrc/up_matmul_sm89.cu",
        "csrc/lm_head_sm89.cu",
    ];

    for f in &watch_files {
        println!("cargo:rerun-if-changed={}", f);
    }
    println!("cargo:rerun-if-changed=build.rs");

    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(vec![source.to_string()])
        .watch(watch_files.iter().map(|s| s.to_string()))
        .include_path(tk_include)
        .include_path(tk_prototype)
        .include_path("csrc") // for llama_sm89.cuh and op .cu files
        .arg("-std=c++20")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-extended-lambda")
        .arg("--expt-relaxed-constexpr")
        .arg("-DKITTENS_4090")
        .arg("-DNDEBUG")
        .arg("-Xcompiler=-fPIC")
        .arg("-Xcompiler=-fno-strict-aliasing")
        .arg("-Xcompiler=-Wno-psabi")
        .arg("-arch=sm_89")
        .arg("-Xptxas=--warn-on-spills")
        .arg("-Xptxas=--verbose")
        .arg("-Xnvlink=--verbose")
        .arg("-lineinfo")
        .build_lib(format!("{}/libtk_llama.a", cache_str))
        .expect("Failed to build tk_llama kernel");

    println!("cargo:rustc-link-search={}", cache_str);
    println!("cargo:rustc-link-lib=static=tk_llama");

    // Find CUDA toolkit lib path for cudart_static.
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rustc-link-search={}/lib64", cuda_path);
    println!("cargo:rustc-link-search={}/lib", cuda_path);

    // TK kernel uses cublas for nothing, but cudart is needed for launch.
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=cuda");
}
