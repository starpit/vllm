// SPDX-License-Identifier: Apache-2.0
// Links the CUDA kernel .a files compiled by ferrite-cuda-builder's build.rs.
// Kernel compilation lives in that crate; this build.rs only emits linker flags.

fn main() {
    #[cfg(feature = "cuda")]
    cuda_link();

    #[cfg(not(feature = "cuda"))]
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn cuda_link() {
    // Locate the shared cudaforge cache populated by ferrite-cuda-builder's build.rs.
    let cache_dir = dirs::cache_dir()
        .expect("no cache directory found")
        .join("cudaforge")
        .join("vllm-cuda");
    let cache_str = cache_dir.to_string_lossy().to_string();

    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rustc-link-search={}", cache_str);

    // libmegakernels.a — built by ferrite-cuda-builder's build.rs
    // for tk-mvp's KVM interpreter path. Emit the link directive
    // UNCONDITIONALLY: the prior `if mk_lib.exists()` check ran at
    // THIS build.rs's execution time, but cargo can run this
    // build.rs in PARALLEL with ferrite-cuda-builder's build.rs
    // (vllm-cuda has only `dirs` as a [build-dependency], so its
    // build.rs doesn't wait on ferrite-cuda-builder). If libmega.a
    // hadn't been built yet at this moment, the conditional
    // silently dropped `-lmegakernels` from the linker command,
    // and at link time (after libmega.a was built) the wrapper-fn's
    // extern decl referenced an unresolved symbol — even though the
    // symbol was sitting right there in the .a file the linker
    // never searched.
    //
    // Always-emit. If libmega.a actually doesn't exist at link
    // time (e.g. the user disabled megakernels), the linker fails
    // loudly with a clear "library not found" rather than the
    // confusing undefined-symbol-from-a-defined-symbol pattern.
    println!("cargo:rustc-link-lib=static=megakernels");

    println!("cargo:rustc-link-lib=static=vllm_kernels");
    println!("cargo:rustc-link-lib=static=ggml_kernels");
    println!("cargo:rustc-link-lib=static=marlin_kernels");
    println!("cargo:rustc-link-lib=static=marlin_moe_kernels");
    println!("cargo:rustc-link-lib=static=cutlass_scaled_mm");
    println!("cargo:rustc-link-lib=static=cutlass_standalone_gemm");
    println!("cargo:rustc-link-lib=static=cutlass_gemm_silu_mul");
    println!("cargo:rustc-link-lib=static=cutlass_gemm_bias");
    println!("cargo:rustc-link-lib=static=vllm_flash_attn");
    println!("cargo:rustc-link-lib=static=flashinfer_attn");

    // cudart_static requires rt + dl; cublas/cublasLt remain dynamic.
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
