// SPDX-License-Identifier: Apache-2.0
// Build script: compile CUDA kernels for the vllm-cuda backend.
//
// Compiles:
// 1. Custom kernels from vllm-kernels/csrc/ (norm, activation, RoPE, embedding, cache)
// 2. FlashAttention-2 kernels from third_party/candle-flash-attn/kernels/
//
// Zero dependency on candle or vllm-kernels at the Rust level.

fn main() {
    #[cfg(feature = "cuda")]
    cuda_build();
}

#[cfg(feature = "cuda")]
fn cuda_build() {
    let kernels_csrc = std::path::Path::new("../vllm-kernels/csrc");

    // 1. vllm-cuda-only kernels (embedding gather, split_qkv).
    // Shared kernels (norm, activation, RoPE, cache, sampling) come from
    // vllm-kernels via its `cuda` feature — we link against that crate's
    // compiled static lib to avoid duplicate symbol errors.
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-gencode=arch=compute_80,code=sm_80")
        .flag("-gencode=arch=compute_86,code=sm_86")
        .flag("-gencode=arch=compute_89,code=sm_89")
        .flag("-gencode=arch=compute_90,code=sm_90")
        .flag("-O3")
        .flag("--use_fast_math")
        .include(kernels_csrc)
        .file(kernels_csrc.join("embedding_kernels.cu"));
    build.compile("vllm_cuda_kernels");

    println!("cargo:rerun-if-changed=../vllm-kernels/csrc/embedding_kernels.cu");

    // 2. FlashAttention-2 paged kernels.
    build_flash_attention();
}

#[cfg(feature = "cuda")]
fn build_flash_attention() {
    use candle_flash_attn_build::{cutlass_include_arg, fetch_cutlass};
    use std::path::PathBuf;

    let fa_dir = std::path::Path::new("../../third_party/candle-flash-attn");

    const CUTLASS_COMMIT: &str = "7d49e6c7e2f8896c47f586706e67e1fb215529dc";

    let kernel_files: Vec<&str> = vec![
        "kernels/flash_api.cu",
        "kernels/flash_fwd_hdim128_fp16_sm80.cu",
        "kernels/flash_fwd_hdim160_fp16_sm80.cu",
        "kernels/flash_fwd_hdim192_fp16_sm80.cu",
        "kernels/flash_fwd_hdim224_fp16_sm80.cu",
        "kernels/flash_fwd_hdim256_fp16_sm80.cu",
        "kernels/flash_fwd_hdim32_fp16_sm80.cu",
        "kernels/flash_fwd_hdim64_fp16_sm80.cu",
        "kernels/flash_fwd_hdim96_fp16_sm80.cu",
        "kernels/flash_fwd_hdim128_bf16_sm80.cu",
        "kernels/flash_fwd_hdim160_bf16_sm80.cu",
        "kernels/flash_fwd_hdim192_bf16_sm80.cu",
        "kernels/flash_fwd_hdim224_bf16_sm80.cu",
        "kernels/flash_fwd_hdim256_bf16_sm80.cu",
        "kernels/flash_fwd_hdim32_bf16_sm80.cu",
        "kernels/flash_fwd_hdim64_bf16_sm80.cu",
        "kernels/flash_fwd_hdim96_bf16_sm80.cu",
        "kernels/flash_fwd_hdim128_fp16_causal_sm80.cu",
        "kernels/flash_fwd_hdim160_fp16_causal_sm80.cu",
        "kernels/flash_fwd_hdim192_fp16_causal_sm80.cu",
        "kernels/flash_fwd_hdim224_fp16_causal_sm80.cu",
        "kernels/flash_fwd_hdim256_fp16_causal_sm80.cu",
        "kernels/flash_fwd_hdim32_fp16_causal_sm80.cu",
        "kernels/flash_fwd_hdim64_fp16_causal_sm80.cu",
        "kernels/flash_fwd_hdim96_fp16_causal_sm80.cu",
        "kernels/flash_fwd_hdim128_bf16_causal_sm80.cu",
        "kernels/flash_fwd_hdim160_bf16_causal_sm80.cu",
        "kernels/flash_fwd_hdim192_bf16_causal_sm80.cu",
        "kernels/flash_fwd_hdim224_bf16_causal_sm80.cu",
        "kernels/flash_fwd_hdim256_bf16_causal_sm80.cu",
        "kernels/flash_fwd_hdim32_bf16_causal_sm80.cu",
        "kernels/flash_fwd_hdim64_bf16_causal_sm80.cu",
        "kernels/flash_fwd_hdim96_bf16_causal_sm80.cu",
    ];

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let build_dir = match std::env::var("CANDLE_FLASH_ATTN_BUILD_DIR") {
        Err(_) => out_dir.clone(),
        Ok(d) => PathBuf::from(d)
            .canonicalize()
            .expect("FA build dir missing"),
    };

    let cutlass_dir = fetch_cutlass(&out_dir, CUTLASS_COMMIT).expect("fetch cutlass");
    let cutlass_include: &'static str =
        Box::leak(cutlass_include_arg(&cutlass_dir).into_boxed_str());

    let kernel_paths: Vec<PathBuf> = kernel_files.iter().map(|f| fa_dir.join(f)).collect();

    let mut builder = bindgen_cuda::Builder::default()
        .kernel_paths(kernel_paths)
        .out_dir(build_dir.clone())
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg(cutlass_include)
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--verbose");

    if let Ok(target) = std::env::var("TARGET")
        && target.contains("msvc")
    {
        builder = builder.arg("-D_USE_MATH_DEFINES");
    }
    if !std::env::var("TARGET")
        .map(|t| t.contains("msvc"))
        .unwrap_or(false)
    {
        builder = builder.arg("-Xcompiler").arg("-fPIC");
    }

    let out_file = build_dir.join("libflashattention.a");
    builder.build_lib(out_file);

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=flashattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    for f in &kernel_files {
        println!("cargo:rerun-if-changed={}", fa_dir.join(f).display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        fa_dir.join("kernels/flash_fwd_kernel.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        fa_dir.join("kernels/flash.h").display()
    );
}
