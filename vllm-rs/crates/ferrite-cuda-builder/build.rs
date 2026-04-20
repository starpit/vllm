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

    // 7. Megakernels — .cu files generated by forward!() in ferrite-models.
    //    The dependency on ferrite-models ensures those .cu files exist by now.
    build_megakernels(&cache_str, &mut rerun_files);

    // 8. ferrite-stencil kernels. v1 = a smoke kernel that exercises the
    //    SM89 prelude header (`stencil_prelude_sm89.cuh`) so nvcc catches
    //    regressions in the helpers the emitter references. Step 3b of
    //    the stencil plan swaps this for a hand-picked FA2 reference;
    //    step 3c cuts over to the emitter's generated source.
    build_stencil_kernels(&cache_str, &mut rerun_files);

    // 9. Kittens-based megakernel. Written by ferrite-forward-macro
    //    to ~/.cache/cudaforge/kittens/<model>.cu. Compiled on sm_90a+
    //    against ThunderKittens (env var THUNDERKITTENS_ROOT; falls back
    //    to ~/ThunderKittens).
    build_kittens_kernels(&cache_str, &mut rerun_files);

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
    const CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

    let mut mk_builder = cudaforge::KernelBuilder::new();
    mk_builder = mk_builder
        .out_dir(cache_dir)
        .source_files(megakernel_cus.clone())
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
        .expect("failed to build megakernel .cu files");

    for cu in &megakernel_cus {
        rerun_files.push(cu.clone());
    }
}

#[cfg(feature = "cuda")]
fn build_stencil_kernels(cache_dir: &str, rerun_files: &mut Vec<String>) {
    // SM89 only at step 3a; SM90 variant lands at step 3e. We gate on
    // the detected arch so the build doesn't fail on H100 machines
    // until the SM90 prelude is in tree.
    let arch = detect_cuda_arch();
    let arch_num: u32 = arch.parse().unwrap_or(89);
    if arch_num >= 90 {
        return;
    }

    let sources = ["../../crates/ferrite-stencil/csrc/stencil_smoke_sm89.cu".to_string()];
    let watch = ["../../crates/ferrite-stencil/csrc/stencil_prelude_sm89.cuh".to_string()];

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(sources.iter().cloned())
        .watch(watch.iter().cloned())
        .include_path("../../crates/ferrite-stencil/csrc")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("-std=c++17")
        .arg("-Xcompiler=-fPIC")
        .arg(&format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
        .arg("-lineinfo")
        .build_lib(format!("{cache_dir}/libstencil_kernels.a"))
        .expect("failed to build ferrite-stencil kernels");

    rerun_files.extend(sources.iter().cloned());
    rerun_files.extend(watch.iter().cloned());
}

#[cfg(feature = "cuda")]
fn build_kittens_kernels(cache_dir: &str, rerun_files: &mut Vec<String>) {
    // SM90a+ only — kittens::tma::load_async + warpgroup::mma_async
    // both require Hopper. Loud diagnostics via `cargo:warning=` so
    // "why didn't kittens_kernels build" is visible without -v.
    let arch = detect_cuda_arch();
    let arch_num: u32 = arch.parse().unwrap_or(89);
    println!("cargo:warning=ferrite-cuda-builder: kittens arch={arch} (num={arch_num})");
    // Re-run build.rs when the CUDA_ARCH env var changes so switching
    // between sm89/sm90 boxes invalidates the cache.
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=THUNDERKITTENS_ROOT");
    if arch_num < 90 {
        println!(
            "cargo:warning=ferrite-cuda-builder: arch<90 ({arch_num}), skipping \
             libkittens_kernels.a. Set CUDA_ARCH=90 to force it on if nvidia-smi \
             isn't reporting compute_cap correctly.",
        );
        return;
    }

    // Inputs: per-model .cu files written by the `#[forward]` macro
    // in ferrite-forward-macro (calls ferrite_stencil::emit_kittens).
    let kittens_cache = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("cudaforge/kittens");

    let kittens_cus: Vec<String> = if kittens_cache.exists() {
        std::fs::read_dir(&kittens_cache)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "cu"))
            .map(|e| e.path().display().to_string())
            .collect()
    } else {
        vec![]
    };

    println!(
        "cargo:warning=ferrite-cuda-builder: found {} kittens .cu files in {:?}",
        kittens_cus.len(),
        kittens_cache,
    );
    if kittens_cus.is_empty() {
        println!(
            "cargo:warning=ferrite-cuda-builder: no kittens .cu files; skipping \
             libkittens_kernels.a. Touch a ferrite-model-* lib.rs to force the \
             `#[forward]` macro to regenerate the .cu cache.",
        );
        return;
    }

    // ThunderKittens include path. Respect THUNDERKITTENS_ROOT env
    // var; fall back to ~/ThunderKittens. Fail the build with a
    // clear message if neither works — TK is a hard dependency on
    // sm_90a+, not optional.
    let tk_root = std::env::var("THUNDERKITTENS_ROOT").unwrap_or_else(|_| {
        dirs::home_dir()
            .expect("no home dir")
            .join("ThunderKittens")
            .to_string_lossy()
            .into_owned()
    });
    let tk_include = format!("{tk_root}/include");
    println!(
        "cargo:warning=ferrite-cuda-builder: ThunderKittens include = {tk_include}",
    );
    if !std::path::Path::new(&tk_include).exists() {
        panic!(
            "ThunderKittens not found at {tk_include}. Set THUNDERKITTENS_ROOT \
             or clone https://github.com/HazyResearch/ThunderKittens to ~/ThunderKittens.",
        );
    }

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(kittens_cus.clone())
        .include_path(&tk_include)
        // Our own csrc dir — hosts kittens_attn.cuh (the __device__
        // port of TK's fwd_attend_ker) that emit_kittens references.
        .include_path("../../crates/ferrite-stencil/csrc")
        .arg("-DKITTENS_HOPPER")
        .arg("-DNDEBUG")
        .arg("-std=c++20")
        .arg("--expt-extended-lambda")
        .arg("--expt-relaxed-constexpr")
        .arg("--use_fast_math")
        .arg("-Xcompiler=-fPIC")
        .arg("-O3")
        .arg("-lineinfo")
        // SM_90a (the 'a' variant) — wgmma / TMA are sm_90a-only.
        .arg("-gencode=arch=compute_90a,code=sm_90a")
        .build_lib(format!("{cache_dir}/libkittens_kernels.a"))
        .expect("failed to build kittens-based megakernel .cu files");

    println!(
        "cargo:warning=ferrite-cuda-builder: wrote {cache_dir}/libkittens_kernels.a",
    );

    for cu in &kittens_cus {
        rerun_files.push(cu.clone());
    }
    rerun_files.push(tk_include);
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
