// SPDX-License-Identifier: Apache-2.0
// Compiles all CUDA kernels for vllm-rs into static .a files via cudaforge.
// Outputs land in the shared cudaforge cache (~/.cache/cudaforge/vllm-cuda/)
// and are linked by vllm-cuda's build.rs.
//
// Keeping kernel compilation in this separate crate allows Docker to cache
// the (slow) kernel build as a distinct layer that only invalidates when
// .cu source files change, independent of Rust source changes.

// Reuse the same config module the lib crate exports so that symbol
// names emitted downstream match what this build.rs renders.
#[cfg(feature = "cuda")]
#[path = "src/flashinfer_config.rs"]
mod flashinfer_config;

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
        "../../crates/vllm-cuda/csrc/grouped_topk_noaux.cu",
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
        "../../crates/vllm-cuda/csrc/precision_cast_kernels.cu",
    ];
    let vllm_watch = [
        "../../crates/vllm-cuda/csrc/moeTopKFuncs.cuh",
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
        .arg("--expt-extended-lambda")
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
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

    // 5b. FlashInfer per-tuple paged-attention shims. Rendered at build
    //     time from `templates/` into `$OUT_DIR/flashinfer_inst/` and
    //     compiled into `libflashinfer_attn.a`.
    build_flashinfer_attention(&cache_str, &mut rerun_files);

    // 6. CUTLASS standalone GEMM launchers (128×128 + 64×64 for solver dispatch)
    build_cutlass_standalone_gemm(&cache_str, &mut rerun_files);

    // 6b. CUTLASS fused Gate GEMM + SiLU + Mul epilogue (EVT) kernel.
    build_cutlass_gemm_silu_mul(&cache_str, &mut rerun_files);

    // 6c. CUTLASS fused GEMM + bias broadcast epilogue (EVT) kernel.
    build_cutlass_gemm_bias(&cache_str, &mut rerun_files);

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
fn build_flashinfer_attention(cache_dir: &str, rerun_files: &mut Vec<String>) {
    use flashinfer_config::FLASHINFER_CONFIG_SET;
    use minijinja::{Environment, context};

    // Upstream FlashInfer commit the shim + forked planner were written
    // against. Bumping this requires re-reading upstream's
    //   csrc/batch_attention_customize_config.jinja
    //   include/flashinfer/attention/scheduler.cuh  (fork source)
    // and reconciling the template text under `templates/`.
    const FLASHINFER_COMMIT: &str = "08ab45d67705b301ee66e63c6999c934c72dd41c";
    const CONFIG_TEMPLATE: &str = include_str!("templates/batch_attention_config.inc.j2");
    const SHIM_TEMPLATE: &str = include_str!("templates/flashinfer_shim.cu.j2");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let inst_dir = std::path::Path::new(&out_dir).join("flashinfer_inst");
    std::fs::create_dir_all(&inst_dir).expect("create flashinfer_inst dir");

    let mut env = Environment::new();
    env.add_template("cfg", CONFIG_TEMPLATE)
        .expect("add config template");
    env.add_template("shim", SHIM_TEMPLATE)
        .expect("add shim template");

    let mut cu_files: Vec<String> = Vec::new();
    for cfg in FLASHINFER_CONFIG_SET {
        let suffix = cfg.sym_suffix();
        let cfg_filename = format!("config_{}.inc", suffix);
        let cu_filename = format!("flashinfer_shim_{}.cu", suffix);

        let cfg_body = env
            .get_template("cfg")
            .unwrap()
            .render(context! {
                dtype_cpp => cfg.dtype.cpp_ty(),
                head_dim => cfg.head_dim,
                flashinfer_commit => FLASHINFER_COMMIT,
            })
            .expect("render config .inc");
        std::fs::write(inst_dir.join(&cfg_filename), cfg_body).expect("write config .inc");

        let shim_body = env
            .get_template("shim")
            .unwrap()
            .render(context! {
                sym_suffix => suffix,
                use_logits_soft_cap => cfg.use_logits_soft_cap,
                config_inc_filename => cfg_filename,
                flashinfer_commit => FLASHINFER_COMMIT,
            })
            .expect("render shim .cu");
        let cu_path = inst_dir.join(&cu_filename);
        std::fs::write(&cu_path, shim_body).expect("write shim .cu");
        cu_files.push(cu_path.to_string_lossy().into_owned());
    }

    // Template sources — re-render on edit. The rendered files under
    // $OUT_DIR are NOT added; they're regenerated every build and the
    // cudaforge per-object cache avoids recompilation when their content
    // is byte-identical.
    rerun_files.push("src/flashinfer_config.rs".to_string());
    rerun_files.push("templates/batch_attention_config.inc.j2".to_string());
    rerun_files.push("templates/flashinfer_shim.cu.j2".to_string());

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(cu_files)
        .include_path(inst_dir.to_string_lossy().as_ref())
        .with_git_dependency(
            "flashinfer",
            "https://github.com/flashinfer-ai/flashinfer.git",
            FLASHINFER_COMMIT,
            vec!["include"],
            /*recurse_submodules=*/ false,
        )
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
        .build_lib(format!("{}/libflashinfer_attn.a", cache_dir))
        .expect("Failed to build flashinfer_attn");
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
fn build_cutlass_gemm_silu_mul(cache_dir: &str, rerun_files: &mut Vec<String>) {
    // Shares the scaled_mm CUTLASS commit (uses cutlass_2x_gemm + EVT
    // infrastructure from scaled_mm_c2x.cuh).
    const CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

    let sources = vec!["../../crates/vllm-cuda/csrc/cutlass_gemm_silu_mul.cu".to_string()];
    let watch = [
        "../../crates/vllm-cuda/csrc/cutlass_silu_mul_epilogue.hpp",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/scaled_mm_c2x.cuh",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/common.hpp",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/math.hpp",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/scaled_mm_epilogues_c2x.hpp",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/broadcast_load_epilogue_c2x.hpp",
    ];

    rerun_files.extend(sources.iter().cloned());
    rerun_files.extend(watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(sources)
        .watch(watch.iter().map(|s| s.to_string()))
        .include_path("../../crates/vllm-cuda/csrc")
        .include_path("../../crates/vllm-cuda/csrc/cutlass_scaled_mm")
        .with_cutlass(Some(CUTLASS_COMMIT))
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .build_lib(format!("{}/libcutlass_gemm_silu_mul.a", cache_dir))
        .expect("Failed to build cutlass_gemm_silu_mul");
}

#[cfg(feature = "cuda")]
fn build_cutlass_gemm_bias(cache_dir: &str, rerun_files: &mut Vec<String>) {
    const CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

    let sources = vec!["../../crates/vllm-cuda/csrc/cutlass_gemm_bias.cu".to_string()];
    let watch = [
        "../../crates/vllm-cuda/csrc/cutlass_gemm_bias_epilogue.hpp",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/scaled_mm_c2x.cuh",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/common.hpp",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/math.hpp",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/scaled_mm_epilogues_c2x.hpp",
        "../../crates/vllm-cuda/csrc/cutlass_scaled_mm/broadcast_load_epilogue_c2x.hpp",
    ];

    rerun_files.extend(sources.iter().cloned());
    rerun_files.extend(watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(sources)
        .watch(watch.iter().map(|s| s.to_string()))
        .include_path("../../crates/vllm-cuda/csrc")
        .include_path("../../crates/vllm-cuda/csrc/cutlass_scaled_mm")
        .with_cutlass(Some(CUTLASS_COMMIT))
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .build_lib(format!("{}/libcutlass_gemm_bias.a", cache_dir))
        .expect("Failed to build cutlass_gemm_bias");
}

#[cfg(feature = "cuda")]
fn build_megakernels(cache_dir: &str, rerun_files: &mut Vec<String>) {
    // Ferrite-TK megakernels (FERRITE_TK_PLAN.md). Each .cu in
    // the cudaforge cache includes kittens.cuh + the ferrite-
    // owned substrate headers under crates/ferrite-kernels/csrc/
    // tk/. No vendor `megakernels/` include path, no CUTLASS
    // dep — ferrite-owned TK 2.0 code only.
    let megakernel_cache = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("cudaforge/megakernels");

    // Phase C step 2 — `.cu` files are written by the
    // `ferrite-mega-cu-emit` binary, which walks the
    // `inventory::collect!(MegaCanonicalEmit)` registry that the
    // `#[forward]` proc-macro populates and writes each
    // `emit_fn()` output to `<cache>/megakernels/ferrite_<canonical>.cu`.
    //
    // Run that bin BEFORE `cargo build --features cuda` of the
    // workspace:
    //
    //   FERRITE_MODELS=llama-3.2-1b FERRITE_MEGA=1 \
    //     cargo run -p ferrite-mega-cu-emit --features cuda
    //   FERRITE_MODELS=llama-3.2-1b FERRITE_MEGA=1 \
    //     cargo build -p vllm-cli --features cuda
    //
    // The existing `.cu` discovery pass below picks them up. A
    // build-deps invocation directly from this build.rs was
    // attempted but breaks proc-macro crate resolution under
    // resolver = "2" when ferrite-models is in both
    // [dependencies] and [build-dependencies] with cuda features
    // (model crates surface "can't find crate for
    // ferrite_forward_macro"). The standalone bin crate sidesteps
    // that dual-edge feature-resolution issue entirely.

    // The ferrite-forward macro emits an `#error "..."` stub in
    // every `.cu` whose schedule contains an op ferrite-TK can't
    // yet lower. Those stubs are written deliberately so cudaforge's
    // content-hash invalidates them when ops land (see
    // `emit_error_variant` in `ferrite-forward-macro/src/interpreter/
    // mega.rs`) — but nvcc would choke on them. Filter them out at
    // discovery time. A stub is detected by the first 30-ish lines
    // containing the marker; reading the whole file is unnecessary.
    fn is_error_stub(path: &std::path::Path) -> bool {
        let Ok(s) = std::fs::read_to_string(path) else {
            return false;
        };
        // Only scan the head — emitted stubs put the `#error` line
        // early (line ~12). Cap at 4 KiB so a genuine long .cu
        // doesn't pay the full-scan cost.
        let head = &s[..s.len().min(4096)];
        head.contains("#error \"ferrite mega:")
    }
    // If FERRITE_MODELS is set, only compile megakernels for those models.
    // The proc-macro maps "llama-3.2-1b" → file stem "llama_3_2_1b" (dashes→underscores).
    // Without this filter, all cached kernels (including large models like gemma2-27b
    // with intermediate_dim=36864) get compiled, which can take 10+ min per kernel
    // when NCW=16 makes the monolithic kernel too large for the PTX optimizer.
    let model_filter: Option<Vec<String>> = std::env::var("FERRITE_MODELS").ok().map(|v| {
        v.split(',')
            .map(|m| m.trim().replace(['-', '.'], "_"))
            .collect()
    });
    println!("cargo:rerun-if-env-changed=FERRITE_MODELS");

    let megakernel_cus: Vec<String> = if megakernel_cache.exists() {
        std::fs::read_dir(&megakernel_cache)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "cu"))
            .filter(|e| !is_error_stub(&e.path()))
            .filter(|e| {
                let Some(ref models) = model_filter else {
                    return true;
                };
                let stem = e
                    .path()
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .trim_start_matches("ferrite_")
                    .to_string();
                models.iter().any(|m| stem.starts_with(m.as_str()))
            })
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

    // Architecture gate for TK primitives. Ferrite is Hopper-
    // first per the plan; if we ever want sm_89 variants we'll
    // thread KITTENS_AMPERE through explicitly.
    let kittens_arch_flag = if arch_num >= 100 {
        "-DKITTENS_BLACKWELL"
    } else if arch_num >= 90 {
        "-DKITTENS_HOPPER"
    } else {
        "-DKITTENS_AMPERE"
    };

    // Hopper TK primitives (`setmaxnreg`, `wgmma`, `tma::*_async`)
    // require the `sm_90a` architecture extension, not plain
    // `sm_90`. ptxas rejects these instructions on `.target
    // 'sm_90'`. Rewrite the gencode target on Hopper.
    let gencode_arch = if arch_num == 90 {
        "90a".to_string()
    } else {
        arch.clone()
    };

    // Include path: ferrite substrate headers first (so
    // `#include "ferrite_globals.cuh"` resolves to ours), then
    // the vendored TK 2.0 headers for `kittens.cuh`.
    let ferrite_tk_include = "../../crates/ferrite-kernels/csrc/tk";
    let tk_include = "../../third_party/thunderkittens/include";

    let mut mk_builder = cudaforge::KernelBuilder::new();
    mk_builder = mk_builder
        .out_dir(cache_dir)
        .source_files(megakernel_cus.clone())
        .include_path(ferrite_tk_include)
        .include_path(tk_include);
    mk_builder
        .arg(std_flag)
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-extended-lambda")
        .arg("--expt-relaxed-constexpr")
        .arg("-DNDEBUG")
        .arg(kittens_arch_flag)
        .arg("-Xcompiler=-fPIC")
        .arg("-Xcompiler=-fno-strict-aliasing")
        .arg("-Xcompiler=-Wno-psabi")
        .arg(&format!(
            "-gencode=arch=compute_{gencode_arch},code=sm_{gencode_arch}"
        ))
        .arg("-lineinfo")
        .build_lib(format!("{cache_dir}/libmegakernels.a"))
        .expect("failed to build megakernel .cu files");

    for cu in &megakernel_cus {
        rerun_files.push(cu.clone());
    }

    // Rerun if any ferrite substrate header changes.
    for entry in walkdir(ferrite_tk_include) {
        rerun_files.push(entry);
    }
}

#[cfg(feature = "cuda")]
fn walkdir(root: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(root)];
    while let Some(p) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&p) {
            for e in rd.flatten() {
                let path = e.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path
                    .extension()
                    .is_some_and(|ext| ext == "cuh" || ext == "h")
                {
                    out.push(path.display().to_string());
                }
            }
        }
    }
    out
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
