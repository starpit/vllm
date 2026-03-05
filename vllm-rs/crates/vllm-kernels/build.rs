// SPDX-License-Identifier: Apache-2.0
// Build script: compile CUDA kernels via nvcc when the `cuda` feature is enabled.

fn main() {
    #[cfg(feature = "cuda")]
    {
        // -----------------------------------------------------------------
        // Library 1: vllm_kernels (existing fused kernels)
        // -----------------------------------------------------------------
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .flag("-gencode=arch=compute_80,code=sm_80") // Ampere (A100)
            .flag("-gencode=arch=compute_86,code=sm_86") // Ampere (A10/A40)
            .flag("-gencode=arch=compute_89,code=sm_89") // Ada (L40S/RTX 4090)
            .flag("-gencode=arch=compute_90,code=sm_90") // Hopper (H100)
            .flag("-O3")
            .flag("--use_fast_math")
            .file("csrc/layernorm_kernels.cu")
            .file("csrc/activation_kernels.cu")
            .file("csrc/pos_encoding_kernels.cu")
            .file("csrc/cache_kernels.cu")
            .file("csrc/qk_norm_rope_kernels.cu")
            .file("csrc/moe_topk_kernels.cu")
            .file("csrc/moe_align_kernels.cu")
            .file("csrc/sampling_kernels.cu")
            .file("csrc/gptq_dequant_kernels.cu")
            .file("csrc/awq_dequant_kernels.cu");
        build.compile("vllm_kernels");

        println!("cargo:rerun-if-changed=csrc/vec_utils.cuh");
        println!("cargo:rerun-if-changed=csrc/layernorm_kernels.cu");
        println!("cargo:rerun-if-changed=csrc/activation_kernels.cu");
        println!("cargo:rerun-if-changed=csrc/pos_encoding_kernels.cu");
        println!("cargo:rerun-if-changed=csrc/cache_kernels.cu");
        println!("cargo:rerun-if-changed=csrc/qk_norm_rope_kernels.cu");
        println!("cargo:rerun-if-changed=csrc/moe_topk_kernels.cu");
        println!("cargo:rerun-if-changed=csrc/moe_align_kernels.cu");
        println!("cargo:rerun-if-changed=csrc/sampling_kernels.cu");
        println!("cargo:rerun-if-changed=csrc/gptq_dequant_kernels.cu");
        println!("cargo:rerun-if-changed=csrc/awq_dequant_kernels.cu");

        // -----------------------------------------------------------------
        // Library 2: marlin_kernels (Marlin W4A16 fused GEMM)
        // -----------------------------------------------------------------
        let mut marlin = cc::Build::new();
        marlin
            .cuda(true)
            .flag("-gencode=arch=compute_80,code=sm_80")
            .flag("-gencode=arch=compute_86,code=sm_86")
            .flag("-gencode=arch=compute_89,code=sm_89")
            .flag("-gencode=arch=compute_90,code=sm_90")
            .flag("-O3")
            .flag("--use_fast_math")
            .flag("-std=c++17")
            .flag("--expt-relaxed-constexpr") // allow constexpr host fns in device code
            .include("csrc/marlin") // for marlin.cuh, kernel.h, etc.
            .include("csrc") // for core/scalar_type.hpp
            // Entry points + repack
            .file("csrc/marlin/marlin_gemm.cu")
            .file("csrc/marlin/gptq_marlin_repack.cu")
            .file("csrc/marlin/awq_marlin_repack.cu")
            // Kernel instantiations (W4A16 only)
            .file("csrc/marlin/sm80_kernel_float16_u4_float16.cu")
            .file("csrc/marlin/sm80_kernel_bfloat16_u4_bfloat16.cu")
            .file("csrc/marlin/sm80_kernel_float16_u4b8_float16.cu")
            .file("csrc/marlin/sm80_kernel_bfloat16_u4b8_bfloat16.cu");
        marlin.compile("marlin_kernels");

        // Rerun on Marlin source changes
        println!("cargo:rerun-if-changed=csrc/marlin/marlin_gemm.cu");
        println!("cargo:rerun-if-changed=csrc/marlin/gptq_marlin_repack.cu");
        println!("cargo:rerun-if-changed=csrc/marlin/awq_marlin_repack.cu");
        println!("cargo:rerun-if-changed=csrc/marlin/marlin.cuh");
        println!("cargo:rerun-if-changed=csrc/marlin/kernel.h");
        println!("cargo:rerun-if-changed=csrc/marlin/kernel_selector.h");
        println!("cargo:rerun-if-changed=csrc/marlin/marlin_template.h");
        println!("cargo:rerun-if-changed=csrc/marlin/marlin_mma.h");
        println!("cargo:rerun-if-changed=csrc/marlin/dequant.h");
        println!("cargo:rerun-if-changed=csrc/marlin/marlin_dtypes.cuh");
        println!("cargo:rerun-if-changed=csrc/core/scalar_type.hpp");
        println!("cargo:rerun-if-changed=csrc/marlin/sm80_kernel_float16_u4_float16.cu");
        println!("cargo:rerun-if-changed=csrc/marlin/sm80_kernel_bfloat16_u4_bfloat16.cu");
        println!("cargo:rerun-if-changed=csrc/marlin/sm80_kernel_float16_u4b8_float16.cu");
        println!("cargo:rerun-if-changed=csrc/marlin/sm80_kernel_bfloat16_u4b8_bfloat16.cu");
    }
}
