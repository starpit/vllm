// SPDX-License-Identifier: Apache-2.0
// Compiles all CUDA kernels for vllm-rs into static .a files via cudaforge.
// Outputs land in the shared cudaforge cache (~/.cache/cudaforge/vllm-cuda/)
// and are linked by vllm-cuda's build.rs.
//
// Keeping kernel compilation in this separate crate allows Docker to cache
// the (slow) kernel build as a distinct layer that only invalidates when
// .cu source files change, independent of Rust source changes.

fn main() {
    #[cfg(feature = "cuda")]
    cuda_build();

    #[cfg(not(feature = "cuda"))]
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn cuda_build() {
    let mut rerun_files: Vec<String> = vec!["build.rs".to_string()];

    // Use the same fixed cache dir as vllm-cuda so the .a files are found
    // when vllm-cuda emits its rustc-link-search directive.
    let cache_dir = dirs::cache_dir()
        .expect("no cache directory found")
        .join("cudaforge")
        .join("vllm-cuda");
    std::fs::create_dir_all(&cache_dir).expect("Failed to create cudaforge cache dir");
    let cache_str = cache_dir.to_string_lossy().to_string();

    // 1. vllm fused kernels
    let vllm_sources = [
        "../../crates/vllm-cuda/csrc/layernorm_kernels.cu",
        "../../crates/vllm-cuda/csrc/activation_kernels.cu",
        "../../crates/vllm-cuda/csrc/pos_encoding_kernels.cu",
        "../../crates/vllm-cuda/csrc/cache_kernels.cu",
        "../../crates/vllm-cuda/csrc/qk_norm_rope_kernels.cu",
        "../../crates/vllm-cuda/csrc/moe_topk_kernels.cu",
        "../../crates/vllm-cuda/csrc/moe_align_kernels.cu",
        "../../crates/vllm-cuda/csrc/moe_align_block_size_kernels.cu",
        "../../crates/vllm-cuda/csrc/fused_moe_gemm_kernels.cu",
        "../../crates/vllm-cuda/csrc/moe_ops_kernels.cu",
        "../../crates/vllm-cuda/csrc/sampling_kernels.cu",
        "../../crates/vllm-cuda/csrc/gptq_dequant_kernels.cu",
        "../../crates/vllm-cuda/csrc/awq_dequant_kernels.cu",
        "../../crates/vllm-cuda/csrc/embedding_kernels.cu",
        "../../crates/vllm-cuda/csrc/bnb_dequant_kernels.cu",
        "../../crates/vllm-cuda/csrc/gdn_gating_kernels.cu",
        "../../crates/vllm-cuda/csrc/gdn_conv1d_kernels.cu",
        "../../crates/vllm-cuda/csrc/gdn_recurrent_kernels.cu",
        "../../crates/vllm-cuda/csrc/gdn_split_kernels.cu",
        "../../crates/vllm-cuda/csrc/mla_kernels.cu",
        "../../crates/vllm-cuda/csrc/dequant_gather_pages.cu",
        "../../crates/vllm-cuda/csrc/fp8_scale_kernels.cu",
        "../../crates/vllm-cuda/csrc/fp8_quant_kernels.cu",
        "../../crates/vllm-cuda/csrc/fp8_block_dequant_kernels.cu",
        "../../crates/vllm-cuda/csrc/fp8_post_scale_kernels.cu",
        "../../crates/vllm-cuda/csrc/gather_last_dim_kernel.cu",
    ];
    let vllm_watch = [
        "../../crates/vllm-cuda/csrc/vec_utils.cuh",
        "../../crates/vllm-cuda/csrc/fp8_utils.cuh",
    ];

    rerun_files.extend(vllm_sources.iter().map(|s| s.to_string()));
    rerun_files.extend(vllm_watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(vllm_sources.iter().map(|s| s.to_string()))
        .watch(vllm_watch.iter().map(|s| s.to_string()))
        .include_path("../../crates/vllm-cuda/csrc")
        .arg("-O3")
        .arg("--use_fast_math")
        .build_lib(format!("{}/libvllm_kernels.a", cache_str))
        .expect("Failed to build vllm_kernels");

    // 2. GGML quantized kernels
    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(vec!["../../crates/vllm-cuda/csrc/quantized.cu".to_string()])
        .arg("-O3")
        .arg("--use_fast_math")
        .build_lib(format!("{}/libggml_kernels.a", cache_str))
        .expect("Failed to build ggml_kernels");

    // 3. Marlin W4A16 fused GEMM kernels
    let marlin_sources = [
        "../../crates/vllm-cuda/csrc/marlin/marlin_gemm.cu",
        "../../crates/vllm-cuda/csrc/marlin/gptq_marlin_repack.cu",
        "../../crates/vllm-cuda/csrc/marlin/awq_marlin_repack.cu",
        "../../crates/vllm-cuda/csrc/marlin/sm80_kernel_float16_u4_float16.cu",
        "../../crates/vllm-cuda/csrc/marlin/sm80_kernel_bfloat16_u4_bfloat16.cu",
        "../../crates/vllm-cuda/csrc/marlin/sm80_kernel_float16_u4b8_float16.cu",
        "../../crates/vllm-cuda/csrc/marlin/sm80_kernel_bfloat16_u4b8_bfloat16.cu",
    ];
    let marlin_watch = [
        "../../crates/vllm-cuda/csrc/marlin/marlin.cuh",
        "../../crates/vllm-cuda/csrc/marlin/kernel.h",
        "../../crates/vllm-cuda/csrc/marlin/kernel_selector.h",
        "../../crates/vllm-cuda/csrc/marlin/marlin_template.h",
        "../../crates/vllm-cuda/csrc/marlin/marlin_mma.h",
        "../../crates/vllm-cuda/csrc/marlin/dequant.h",
        "../../crates/vllm-cuda/csrc/marlin/marlin_dtypes.cuh",
        "../../crates/vllm-cuda/csrc/core/scalar_type.hpp",
    ];

    rerun_files.extend(marlin_sources.iter().map(|s| s.to_string()));
    rerun_files.extend(marlin_watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(marlin_sources.iter().map(|s| s.to_string()))
        .watch(marlin_watch.iter().map(|s| s.to_string()))
        .include_path("../../crates/vllm-cuda/csrc/marlin")
        .include_path("../../crates/vllm-cuda/csrc")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        .build_lib(format!("{}/libmarlin_kernels.a", cache_str))
        .expect("Failed to build marlin_kernels");

    // 3b. Marlin MoE kernels
    let marlin_moe_sources = [
        "../../crates/vllm-cuda/csrc/marlin_moe/marlin_moe_gemm.cu",
        "../../crates/vllm-cuda/csrc/marlin_moe/sm80_kernel_float16_u4_float16.cu",
        "../../crates/vllm-cuda/csrc/marlin_moe/sm80_kernel_bfloat16_u4_bfloat16.cu",
        "../../crates/vllm-cuda/csrc/marlin_moe/sm80_kernel_float16_u4b8_float16.cu",
        "../../crates/vllm-cuda/csrc/marlin_moe/sm80_kernel_bfloat16_u4b8_bfloat16.cu",
    ];
    let marlin_moe_watch = [
        "../../crates/vllm-cuda/csrc/marlin_moe/kernel.h",
        "../../crates/vllm-cuda/csrc/marlin_moe/kernel_selector.h",
        "../../crates/vllm-cuda/csrc/marlin_moe/marlin_template.h",
    ];

    rerun_files.extend(marlin_moe_sources.iter().map(|s| s.to_string()));
    rerun_files.extend(marlin_moe_watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(marlin_moe_sources.iter().map(|s| s.to_string()))
        .watch(marlin_moe_watch.iter().map(|s| s.to_string()))
        .include_path("../../crates/vllm-cuda/csrc/marlin_moe")
        .include_path("../../crates/vllm-cuda/csrc/marlin")
        .include_path("../../crates/vllm-cuda/csrc")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        .build_lib(format!("{}/libmarlin_moe_kernels.a", cache_str))
        .expect("Failed to build marlin_moe_kernels");

    // 4. CUTLASS scaled_mm FP8 GEMM kernels
    build_cutlass_scaled_mm(&cache_str, &mut rerun_files);

    // 5. FlashAttention-2 paged kernels
    build_flash_attention(&cache_str, &mut rerun_files);

    // 6. CUTLASS standalone GEMM launchers (128×128 + 64×64 for solver dispatch)
    build_cutlass_standalone_gemm(&cache_str, &mut rerun_files);

    // 7. Megakernels — .cu files generated by forward!() in ferrite-models.
    //    The dependency on ferrite-models ensures those .cu files exist by now.
    build_megakernels(&cache_str, &mut rerun_files);

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
    const CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

    let scaled_mm_sources =
        vec!["../../crates/vllm-cuda/csrc/cutlass_scaled_mm/scaled_mm_c2x_sm89.cu".to_string()];
    let scaled_mm_watch = [
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/common.hpp",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/math.hpp",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/scaled_mm_c2x.cuh",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/scaled_mm_c2x_sm89_fp8_dispatch.cuh",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/scaled_mm_epilogues_c2x.hpp",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/broadcast_load_epilogue_c2x.hpp",
    ];

    rerun_files.extend(scaled_mm_sources.iter().cloned());
    rerun_files.extend(scaled_mm_watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(scaled_mm_sources)
        .watch(scaled_mm_watch.iter().map(|s| s.to_string()))
        .include_path("../../crates/vllm-cuda/csrc/cutlass_scaled_mm")
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
}

#[cfg(feature = "cuda")]
fn build_flash_attention(cache_dir: &str, rerun_files: &mut Vec<String>) {
    let fa_src = std::path::Path::new("../../third_party/vllm-flash-attn/src");
    let shim_dir = std::path::Path::new("../../third_party/flash-attn-shim");

    const CUTLASS_COMMIT: &str = "62750a2b75c802660e4894434dc55e839f322277";

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
    kernel_files.push(shim_dir.join("ffi_shim.cu").to_string_lossy().into_owned());

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
}

#[cfg(feature = "cuda")]
fn build_cutlass_standalone_gemm(cache_dir: &str, rerun_files: &mut Vec<String>) {
    // Same CUTLASS commit as scaled_mm — the standalone GEMM only uses
    // the CUTLASS 2.x device::Gemm interface, compatible with any recent commit.
    const CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

    let sources = vec!["../../crates/vllm-cuda/csrc/cutlass_standalone_gemm.cu".to_string()];
    rerun_files.extend(sources.iter().cloned());

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(sources)
        .with_cutlass(Some(CUTLASS_COMMIT))
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-relaxed-constexpr")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .build_lib(format!("{}/libcutlass_standalone_gemm.a", cache_dir))
        .expect("Failed to build cutlass_standalone_gemm");
}

#[cfg(feature = "cuda")]
fn build_megakernels(cache_dir: &str, rerun_files: &mut Vec<String>) {
    let megakernel_cache = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("cudaforge/megakernels");

    let megakernel_cus: Vec<String> = if megakernel_cache.exists() {
        std::fs::read_dir(&megakernel_cache)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "cu"))
            .map(|e| e.path().display().to_string())
            .collect()
    } else {
        vec![]
    };

    if megakernel_cus.is_empty() {
        return;
    }

    let arch = detect_cuda_arch();
    let arch_num: u32 = arch.parse().unwrap_or(89);
    let std_flag = if arch_num >= 90 {
        "-std=c++20"
    } else {
        "-std=c++17"
    };

    // The megakernel .cu files include megakernel_ops.cuh from vllm-cuda/csrc.
    // TK megakernels (SM90+) additionally include kittens.cuh and the KVM runtime.
    const CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

    // Separate TK megakernels (files containing "tk_megakernel") from BSP ones.
    let (tk_cus, bsp_cus): (Vec<String>, Vec<String>) = megakernel_cus
        .iter()
        .cloned()
        .partition(|f| f.contains("tk_megakernel"));

    // Build BSP megakernels (sm89+).
    if !bsp_cus.is_empty() {
        let mut mk_builder = cudaforge::KernelBuilder::new();
        mk_builder = mk_builder
            .out_dir(cache_dir)
            .source_files(bsp_cus.clone())
            .include_path("../../crates/vllm-cuda/csrc")
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
            .build_lib(format!("{cache_dir}/libmegakernels.a"))
            .expect("failed to build BSP megakernel .cu files");
    }

    // Build TK megakernels (sm90+ only, requires ThunderKittens + KVM runtime).
    // The vendored ops only support specific model configs (head_dim, GQA ratio).
    // Build each .cu individually and collect only those that succeed.
    if !tk_cus.is_empty() && arch_num >= 90 {
        let mut ok_cus: Vec<String> = Vec::new();
        for cu in &tk_cus {
            let mut tk_builder = cudaforge::KernelBuilder::new();
            tk_builder = tk_builder
                .out_dir(cache_dir)
                .source_files(vec![cu.clone()])
                .include_path("../../crates/vllm-cuda/csrc")
                .include_path("../../third_party/ThunderKittens")
                .include_path("../../third_party/Megakernels/include")
                .include_path("../../third_party/Megakernels/demos/low-latency-llama")
                .with_cutlass(Some(CUTLASS_COMMIT));
            let obj_name = std::path::Path::new(cu)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            let lib_path = format!("{cache_dir}/lib{obj_name}.a");
            let result = tk_builder
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
                // cudaforge auto-adds -gencode=arch=compute_90a,code=sm_90a
                .arg("-lineinfo")
                .build_lib(&lib_path);
            match result {
                Ok(()) => {
                    println!("cargo:warning=TK megakernel OK: {obj_name}");
                    ok_cus.push(lib_path);
                }
                Err(e) => {
                    println!("cargo:warning=TK megakernel SKIPPED (incompatible config): {obj_name}: {e}");
                }
            }
        }
        // Merge successful .a files into a single libtk_megakernels.a
        if !ok_cus.is_empty() {
            let tk_lib = format!("{cache_dir}/libtk_megakernels.a");
            let _ = std::fs::remove_file(&tk_lib);
            let mut ar = std::process::Command::new("ar");
            ar.arg("rcs").arg(&tk_lib);
            for lib in &ok_cus {
                // Extract objects from each individual .a and add to merged archive
                let extract_dir = format!("{cache_dir}/tk_extract");
                let _ = std::fs::create_dir_all(&extract_dir);
                let _ = std::process::Command::new("ar")
                    .arg("x")
                    .arg(lib)
                    .current_dir(&extract_dir)
                    .status();
                // Add all .o files from extraction
                if let Ok(entries) = std::fs::read_dir(&extract_dir) {
                    for entry in entries.flatten() {
                        if entry.path().extension().is_some_and(|e| e == "o") {
                            ar.arg(entry.path());
                        }
                    }
                }
            }
            ar.status().expect("failed to create libtk_megakernels.a");
            // Cleanup
            let _ = std::fs::remove_dir_all(format!("{cache_dir}/tk_extract"));
        }
    }

    for cu in &megakernel_cus {
        rerun_files.push(cu.clone());
    }
}

#[cfg(feature = "cuda")]
fn detect_cuda_arch() -> String {
    // Try CUDA_ARCH env var first, then probe via nvidia-smi.
    if let Ok(arch) = std::env::var("CUDA_ARCH") {
        return arch;
    }
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let cap = String::from_utf8_lossy(&out.stdout);
            let cap = cap.trim().lines().next().unwrap_or("8.9");
            cap.replace('.', "")
        }
        _ => "89".to_string(), // Default to sm89 (Ada Lovelace)
    }
}
