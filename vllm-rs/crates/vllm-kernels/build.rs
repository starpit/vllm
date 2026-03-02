// SPDX-License-Identifier: Apache-2.0
// Build script: compile CUDA kernels via nvcc when the `cuda` feature is enabled.

fn main() {
    #[cfg(feature = "cuda")]
    {
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
            .file("csrc/qk_norm_rope_kernels.cu");
        build.compile("vllm_kernels");

        println!("cargo:rerun-if-changed=csrc/layernorm_kernels.cu");
        println!("cargo:rerun-if-changed=csrc/activation_kernels.cu");
        println!("cargo:rerun-if-changed=csrc/pos_encoding_kernels.cu");
        println!("cargo:rerun-if-changed=csrc/cache_kernels.cu");
        println!("cargo:rerun-if-changed=csrc/qk_norm_rope_kernels.cu");
    }
}
