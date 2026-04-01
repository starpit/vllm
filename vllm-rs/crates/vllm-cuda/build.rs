// SPDX-License-Identifier: Apache-2.0
// Build script: compile ALL CUDA kernels for the vllm-cuda backend.
// Uses cudaforge for incremental builds (only recompiles changed .cu files).
//
// Compiles:
// 1. vllm fused kernels (norm, activation, RoPE, cache, sampling, MoE, quantize, embedding)
// 2. Marlin W4A16 fused GEMM kernels
// 3. FlashAttention-2 paged kernels (vllm-project fork)

fn main() {
    #[cfg(feature = "cuda")]
    cuda_build();

    #[cfg(not(feature = "cuda"))]
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn cuda_build() {
    // cudarc's static-linking feature emits `rustc-link-lib=static:+whole-archive=stdc++`
    // but doesn't add the GCC-versioned lib directory to the search path. Detect it here
    // so the linker can find libstdc++.a regardless of GCC version or distro layout.
    // cudarc's static-linking feature emits `rustc-link-lib=static:+whole-archive=stdc++`
    // but doesn't add the GCC versioned lib directory. Use rustc-flags (not
    // rustc-link-search) so the -L propagates to all crates including cudarc itself.
    if let Ok(output) = std::process::Command::new("gcc")
        .arg("-print-file-name=libstdc++.a")
        .output()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        // gcc echoes the input unchanged when the file isn't found
        if path != "libstdc++.a" {
            if let Some(dir) = std::path::Path::new(&path).parent() {
                println!("cargo:rustc-flags=-L {}", dir.display());
            }
        }
    }

    // Track all source/header files for cargo:rerun-if-changed.
    // Without these directives, cargo re-runs build.rs on EVERY build,
    // marking vllm-cuda dirty and forcing recompilation of all downstream crates.
    let mut rerun_files: Vec<String> = vec!["build.rs".to_string()];

    // Use a stable shared cache directory so clippy/test/build reuse compiled .o files.
    // Cargo gives each profile a different OUT_DIR, which defeats cudaforge's incremental cache.
    let cache_dir = dirs::cache_dir()
        .expect("no cache directory found")
        .join("cudaforge")
        .join("vllm-cuda");
    std::fs::create_dir_all(&cache_dir).expect("Failed to create cudaforge cache dir");
    let cache_str = cache_dir.to_string_lossy().to_string();

    // 1. vllm fused kernels (merged from vllm-kernels + embedding gather).
    let vllm_sources = [
        "csrc/layernorm_kernels.cu",
        "csrc/activation_kernels.cu",
        "csrc/pos_encoding_kernels.cu",
        "csrc/cache_kernels.cu",
        "csrc/qk_norm_rope_kernels.cu",
        "csrc/moe_topk_kernels.cu",
        "csrc/moe_align_kernels.cu",
        "csrc/moe_align_block_size_kernels.cu",
        "csrc/fused_moe_gemm_kernels.cu",
        "csrc/moe_ops_kernels.cu",
        "csrc/sampling_kernels.cu",
        "csrc/gptq_dequant_kernels.cu",
        "csrc/awq_dequant_kernels.cu",
        "csrc/embedding_kernels.cu",
        "csrc/bnb_dequant_kernels.cu",
        "csrc/gdn_gating_kernels.cu",
        "csrc/gdn_conv1d_kernels.cu",
        "csrc/gdn_recurrent_kernels.cu",
        "csrc/gdn_split_kernels.cu",
        "csrc/mla_kernels.cu",
        "csrc/dequant_gather_pages.cu",
        "csrc/fp8_scale_kernels.cu",
        "csrc/fp8_quant_kernels.cu",
        "csrc/fp8_block_dequant_kernels.cu",
        "csrc/fp8_post_scale_kernels.cu",
        "csrc/gather_last_dim_kernel.cu",
    ];
    let vllm_watch = ["csrc/vec_utils.cuh", "csrc/fp8_utils.cuh"];

    rerun_files.extend(vllm_sources.iter().map(|s| s.to_string()));
    rerun_files.extend(vllm_watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(vllm_sources.iter().map(|s| s.to_string()))
        .watch(vllm_watch.iter().map(|s| s.to_string()))
        .include_path("csrc")
        .arg("-O3")
        .arg("--use_fast_math")
        .build_lib(format!("{}/libvllm_kernels.a", cache_str))
        .expect("Failed to build vllm_kernels");

    println!("cargo:rustc-link-search={}", cache_str);
    println!("cargo:rustc-link-lib=static=vllm_kernels");

    // 2. GGML quantized kernels (llama.cpp-derived, for GGUF inference).
    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(vec!["csrc/quantized.cu".to_string()])
        .arg("-O3")
        .arg("--use_fast_math")
        .build_lib(format!("{}/libggml_kernels.a", cache_str))
        .expect("Failed to build ggml_kernels");

    println!("cargo:rustc-link-lib=static=ggml_kernels");

    // 3. Marlin W4A16 fused GEMM kernels.
    let marlin_sources = [
        "csrc/marlin/marlin_gemm.cu",
        "csrc/marlin/gptq_marlin_repack.cu",
        "csrc/marlin/awq_marlin_repack.cu",
        "csrc/marlin/sm80_kernel_float16_u4_float16.cu",
        "csrc/marlin/sm80_kernel_bfloat16_u4_bfloat16.cu",
        "csrc/marlin/sm80_kernel_float16_u4b8_float16.cu",
        "csrc/marlin/sm80_kernel_bfloat16_u4b8_bfloat16.cu",
    ];
    let marlin_watch = [
        "csrc/marlin/marlin.cuh",
        "csrc/marlin/kernel.h",
        "csrc/marlin/kernel_selector.h",
        "csrc/marlin/marlin_template.h",
        "csrc/marlin/marlin_mma.h",
        "csrc/marlin/dequant.h",
        "csrc/marlin/marlin_dtypes.cuh",
        "csrc/core/scalar_type.hpp",
    ];

    rerun_files.extend(marlin_sources.iter().map(|s| s.to_string()));
    rerun_files.extend(marlin_watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(marlin_sources.iter().map(|s| s.to_string()))
        .watch(marlin_watch.iter().map(|s| s.to_string()))
        .include_path("csrc/marlin")
        .include_path("csrc")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        .build_lib(format!("{}/libmarlin_kernels.a", cache_str))
        .expect("Failed to build marlin_kernels");

    println!("cargo:rustc-link-lib=static=marlin_kernels");

    // 3b. Marlin MoE W4A16 fused GEMM kernels (expert-routed variant).
    let marlin_moe_sources = [
        "csrc/marlin_moe/marlin_moe_gemm.cu",
        "csrc/marlin_moe/sm80_kernel_float16_u4_float16.cu",
        "csrc/marlin_moe/sm80_kernel_bfloat16_u4_bfloat16.cu",
        "csrc/marlin_moe/sm80_kernel_float16_u4b8_float16.cu",
        "csrc/marlin_moe/sm80_kernel_bfloat16_u4b8_bfloat16.cu",
    ];
    let marlin_moe_watch = [
        "csrc/marlin_moe/kernel.h",
        "csrc/marlin_moe/kernel_selector.h",
        "csrc/marlin_moe/marlin_template.h",
    ];

    rerun_files.extend(marlin_moe_sources.iter().map(|s| s.to_string()));
    rerun_files.extend(marlin_moe_watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(marlin_moe_sources.iter().map(|s| s.to_string()))
        .watch(marlin_moe_watch.iter().map(|s| s.to_string()))
        .include_path("csrc/marlin_moe")
        .include_path("csrc/marlin")
        .include_path("csrc")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        .build_lib(format!("{}/libmarlin_moe_kernels.a", cache_str))
        .expect("Failed to build marlin_moe_kernels");

    println!("cargo:rustc-link-lib=static=marlin_moe_kernels");

    // 4. CUTLASS scaled_mm FP8 GEMM kernels (fused per-row scale epilogue).
    build_cutlass_scaled_mm(&cache_str, &mut rerun_files);

    // 5. FlashAttention-2 paged kernels (vllm-project fork).
    build_flash_attention(&cache_str, &mut rerun_files);

    // Emit rerun-if-changed for all tracked files so cargo skips the build
    // script (and all downstream recompilation) when nothing changed.
    // Canonicalize paths so cargo can reliably track them across working
    // directories and worktrees.
    for f in &rerun_files {
        let path = std::path::Path::new(f);
        if let Ok(canonical) = path.canonicalize() {
            println!("cargo:rerun-if-changed={}", canonical.display());
        } else {
            println!("cargo:rerun-if-changed={}", f);
        }
    }
}

#[cfg(feature = "cuda")]
fn build_cutlass_scaled_mm(cache_dir: &str, rerun_files: &mut Vec<String>) {
    // CUTLASS v4.2.1 commit — matches Python vLLM's CUTLASS version.
    // Different from FlashAttention's CUTLASS (which uses the flash-attn fork).
    const CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

    let scaled_mm_sources = vec!["csrc/cutlass_scaled_mm/scaled_mm_c2x_sm89.cu".to_string()];
    let scaled_mm_watch = [
        "csrc/cutlass_scaled_mm/common.hpp",
        "csrc/cutlass_scaled_mm/math.hpp",
        "csrc/cutlass_scaled_mm/scaled_mm_c2x.cuh",
        "csrc/cutlass_scaled_mm/scaled_mm_c2x_sm89_fp8_dispatch.cuh",
        "csrc/cutlass_scaled_mm/scaled_mm_epilogues_c2x.hpp",
        "csrc/cutlass_scaled_mm/broadcast_load_epilogue_c2x.hpp",
    ];

    rerun_files.extend(scaled_mm_sources.iter().cloned());
    rerun_files.extend(scaled_mm_watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(scaled_mm_sources)
        .watch(scaled_mm_watch.iter().map(|s| s.to_string()))
        .include_path("csrc/cutlass_scaled_mm")
        .with_cutlass(Some(CUTLASS_COMMIT))
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .build_lib(format!("{}/libcutlass_scaled_mm.a", cache_dir))
        .expect("Failed to build cutlass_scaled_mm");

    println!("cargo:rustc-link-lib=static=cutlass_scaled_mm");
}

#[cfg(feature = "cuda")]
fn build_flash_attention(cache_dir: &str, rerun_files: &mut Vec<String>) {
    // Upstream kernel source (unchanged from vllm-project/flash-attention)
    let fa_src = std::path::Path::new("../../third_party/vllm-flash-attn/src");
    // Our compat headers (stubs for PyTorch deps) + FFI shim
    let shim_dir = std::path::Path::new("../../third_party/flash-attn-shim");

    // Must match the CUTLASS submodule in vllm-project/flash-attention.
    const CUTLASS_COMMIT: &str = "62750a2b75c802660e4894434dc55e839f322277";

    // Build kernel file list: fwd + splitkv for each headdim/dtype/causal combo
    let hdims = ["32", "64", "96", "128", "192", "256"];
    let mut kernel_files: Vec<String> = Vec::new();
    for hdim in &hdims {
        for suffix in &[
            "fp16_sm80",
            "bf16_sm80",
            "fp16_causal_sm80",
            "bf16_causal_sm80",
        ] {
            kernel_files.push(
                fa_src
                    .join(format!("flash_fwd_hdim{}_{}.cu", hdim, suffix))
                    .to_string_lossy()
                    .into_owned(),
            );
            kernel_files.push(
                fa_src
                    .join(format!("flash_fwd_split_hdim{}_{}.cu", hdim, suffix))
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    // Our FFI shim (the only custom .cu file)
    kernel_files.push(shim_dir.join("ffi_shim.cu").to_string_lossy().into_owned());

    // Header files to watch for changes — include ALL FA2 headers so that
    // any header modification invalidates the cached .o files.
    let watch_files: Vec<String> = vec![
        fa_src.join("flash_fwd_kernel.h"),
        fa_src.join("flash.h"),
        fa_src.join("flash_fwd_launch_template.h"),
        fa_src.join("static_switch.h"),
        fa_src.join("utils.h"),
        fa_src.join("kernel_traits.h"),
        fa_src.join("softmax.h"),
        fa_src.join("mask.h"),
        fa_src.join("rotary.h"),
        fa_src.join("block_info.h"),
        fa_src.join("dropout.h"),
        fa_src.join("namespace_config.h"),
        fa_src.join("philox_unpack.cuh"),
        shim_dir.join("ffi_shim.cu"),
        shim_dir
            .join("compat")
            .join("ATen")
            .join("cuda")
            .join("CUDAGeneratorImpl.h"),
        shim_dir
            .join("compat")
            .join("ATen")
            .join("cuda")
            .join("detail")
            .join("UnpackRaw.cuh"),
        shim_dir
            .join("compat")
            .join("c10")
            .join("cuda")
            .join("CUDAException.h"),
    ]
    .into_iter()
    .map(|p| p.to_string_lossy().into_owned())
    .collect();

    rerun_files.extend(kernel_files.iter().cloned());
    rerun_files.extend(watch_files.iter().cloned());

    // Include order matters: compat stubs FIRST (override PyTorch headers),
    // then upstream kernel source, then CUTLASS (added by with_cutlass).
    let compat_include = shim_dir.join("compat");

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(kernel_files)
        .watch(watch_files)
        .include_path(compat_include.to_string_lossy().as_ref())
        .include_path(fa_src.to_string_lossy().as_ref())
        .with_cutlass(Some(CUTLASS_COMMIT))
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .build_lib(format!("{}/libvllm_flash_attn.a", cache_dir))
        .expect("Failed to build flash attention");

    println!("cargo:rustc-link-lib=static=vllm_flash_attn");
    // rt + dl are required by cudart_static (pulled in transitively by cublas_static).
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
}
