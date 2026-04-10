// SPDX-License-Identifier: Apache-2.0
//! Build script for TK op-level test harness.
//!
//! With `--features cuda`:
//! 1. Generates a standalone CUDA test kernel for each TK op
//! 2. Compiles all into a single .a via cudaforge
//! 3. Links the resulting library

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();
}

#[cfg(feature = "cuda")]
fn build_cuda() {
    use std::path::PathBuf;

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Re-run build.rs when the fused prefill backend selector changes
    println!("cargo:rerun-if-env-changed=TK_FUSED_PREFILL");

    // Set up cudaforge cache directory
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("cudaforge")
        .join("vllm-tk-test-harness");
    std::fs::create_dir_all(&cache_dir).ok();
    let cache_str = cache_dir.display().to_string();

    // Find TK include paths (relative to workspace root)
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let tk_csrc = workspace_root.join("crates/vllm-tk/csrc");
    let tk_include = tk_csrc.join("include");
    let tk_prototype = tk_csrc.join("prototype");

    // Collect TK headers for content-hash tracking
    let header_files: Vec<String> = walkdir(&tk_include)
        .iter()
        .chain(walkdir(&tk_csrc).iter())
        .filter(|p| {
            let s = p.display().to_string();
            s.ends_with(".cuh") || s.ends_with(".cu") || s.ends_with(".h")
        })
        .map(|p| p.display().to_string())
        .collect();

    // DSL for 1B LLaMA (smallest variant, fastest compile)
    let dsl = r#"kernel llama_sm89<NL=16, HD=2048, ID=8192, HDM=64, NAH=32, NKH=8, VS=128256> {
        for layer in 0..NL {
            let normed = rmsnorm(hidden_states, attn_norm[layer]);
            let qkv = gemm(normed, qkv_weights[layer]);
            let (q, k, v) = rope_append(qkv, positions, kv_cache[layer]);
            let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
            hidden_states = gemm_add(attn, o_proj[layer], hidden_states);

            let normed2 = rmsnorm(hidden_states, mlp_norm[layer]);
            let gate = silu(gemm(normed2, gate_weights[layer]));
            let up = gemm(normed2, up_weights[layer]);
            hidden_states = gemm_add(gate * up, down_proj[layer], hidden_states);
        }
        let normed = rmsnorm(hidden_states, lm_head_norm);
        logits = gemm(normed, lm_head);
    }"#;

    // Generate a .cu file for each op
    let mut cu_files = Vec::new();
    for op_name in vllm_tk_macros_core::available_op_names() {
        let cuda_source = vllm_tk_macros_core::generate_single_op_kernel(dsl, op_name)
            .unwrap_or_else(|e| panic!("single-op codegen failed for {op_name}: {e}"));

        let cu_path = out_dir.join(format!("test_{op_name}.cu"));
        std::fs::write(&cu_path, &cuda_source)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", cu_path.display()));
        cu_files.push(cu_path.display().to_string());
    }

    // Generate inline kernels (no KVM protocol)
    let inline_rmsnorm_source = vllm_tk_macros_core::generate_inline_rmsnorm_kernel(dsl)
        .unwrap_or_else(|e| panic!("inline rmsnorm codegen failed: {e}"));
    let inline_rmsnorm_path = out_dir.join("inline_rmsnorm.cu");
    std::fs::write(&inline_rmsnorm_path, &inline_rmsnorm_source)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", inline_rmsnorm_path.display()));
    cu_files.push(inline_rmsnorm_path.display().to_string());

    let inline_gemm_source = vllm_tk_macros_core::generate_inline_gemm_kernel(dsl)
        .unwrap_or_else(|e| panic!("inline gemm codegen failed: {e}"));
    let inline_gemm_path = out_dir.join("inline_gemm.cu");
    std::fs::write(&inline_gemm_path, &inline_gemm_source)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", inline_gemm_path.display()));
    cu_files.push(inline_gemm_path.display().to_string());

    let fused_rmsnorm_gemm_source = vllm_tk_macros_core::generate_fused_rmsnorm_gemm_kernel(dsl)
        .unwrap_or_else(|e| panic!("fused rmsnorm_gemm codegen failed: {e}"));
    let fused_rmsnorm_gemm_path = out_dir.join("fused_rmsnorm_gemm.cu");
    std::fs::write(&fused_rmsnorm_gemm_path, &fused_rmsnorm_gemm_source)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", fused_rmsnorm_gemm_path.display()));
    cu_files.push(fused_rmsnorm_gemm_path.display().to_string());

    let fused_mlp_source = vllm_tk_macros_core::generate_fused_mlp_kernel(dsl)
        .unwrap_or_else(|e| panic!("fused mlp codegen failed: {e}"));
    let fused_mlp_path = out_dir.join("fused_mlp.cu");
    std::fs::write(&fused_mlp_path, &fused_mlp_source)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", fused_mlp_path.display()));
    cu_files.push(fused_mlp_path.display().to_string());

    // CP5-D-4: grid-dispatched fused MLP. The original kernel has
    // hardcoded row=0, layer=0, <<<1,256>>>. Patch to use blockIdx
    // and launch with a grid covering all rows × 1 layer (called
    // per-layer from the host). The kernel body doesn't use barriers
    // (verified: only make_arg<G::barriers> in globals ctor, no
    // reads/writes in the kernel), so grid dispatch is safe.
    let cp5_mlp_source = fused_mlp_source
        .replace("fused_mlp", "cp5_fused_mlp")
        .replace(
            "const int layer = 0;",
            "const int layer = 0; // single-layer dispatch",
        )
        .replace("const int row = 0;", "const int row = blockIdx.x;")
        .replace(
            "cp5_fused_mlp<<<1,",
            "cp5_fused_mlp<<<(batch_size + MLP_BATCH_BLOCK - 1) / MLP_BATCH_BLOCK,",
        );
    let cp5_mlp_path = out_dir.join("cp5_fused_mlp.cu");
    std::fs::write(&cp5_mlp_path, &cp5_mlp_source)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", cp5_mlp_path.display()));
    cu_files.push(cp5_mlp_path.display().to_string());

    let inline_attn_source = vllm_tk_macros_core::generate_inline_attention_decode_kernel(dsl)
        .unwrap_or_else(|e| panic!("inline attention decode codegen failed: {e}"));
    let inline_attn_path = out_dir.join("inline_attention_decode.cu");
    std::fs::write(&inline_attn_path, &inline_attn_source)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", inline_attn_path.display()));
    cu_files.push(inline_attn_path.display().to_string());

    let fused_multi_layer_source = vllm_tk_macros_core::generate_fused_multi_layer_kernel(dsl)
        .unwrap_or_else(|e| panic!("fused multi layer codegen failed: {e}"));
    let fused_multi_layer_path = out_dir.join("fused_multi_layer.cu");
    std::fs::write(&fused_multi_layer_path, &fused_multi_layer_source)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", fused_multi_layer_path.display()));
    cu_files.push(fused_multi_layer_path.display().to_string());

    let fused_full_layer_source = vllm_tk_macros_core::generate_fused_full_layer_kernel(dsl)
        .unwrap_or_else(|e| panic!("fused full layer codegen failed: {e}"));
    let fused_full_layer_path = out_dir.join("fused_full_layer.cu");
    std::fs::write(&fused_full_layer_path, &fused_full_layer_source)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", fused_full_layer_path.display()));
    cu_files.push(fused_full_layer_path.display().to_string());

    let fused_multi_sm_source = vllm_tk_macros_core::generate_fused_multi_sm_kernel(dsl)
        .unwrap_or_else(|e| panic!("fused multi sm codegen failed: {e}"));
    let fused_multi_sm_path = out_dir.join("fused_multi_sm.cu");
    std::fs::write(&fused_multi_sm_path, &fused_multi_sm_source)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", fused_multi_sm_path.display()));
    cu_files.push(fused_multi_sm_path.display().to_string());

    let fused_prefill_source = vllm_tk_macros_core::generate_fused_prefill_kernel(dsl)
        .unwrap_or_else(|e| panic!("fused prefill codegen failed: {e}"));
    let fused_prefill_path = out_dir.join("fused_prefill.cu");
    std::fs::write(&fused_prefill_path, &fused_prefill_source)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", fused_prefill_path.display()));
    cu_files.push(fused_prefill_path.display().to_string());

    let fused_prefill_layer_source = vllm_tk_macros_core::generate_fused_prefill_layer_kernel(dsl)
        .unwrap_or_else(|e| panic!("fused prefill layer codegen failed: {e}"));
    let fused_prefill_layer_path = out_dir.join("fused_prefill_layer.cu");
    std::fs::write(&fused_prefill_layer_path, &fused_prefill_layer_source).unwrap_or_else(|e| {
        panic!(
            "failed to write {}: {e}",
            fused_prefill_layer_path.display()
        )
    });
    cu_files.push(fused_prefill_layer_path.display().to_string());

    let fused_layer_source = vllm_tk_macros_core::generate_fused_layer_kernel(dsl)
        .unwrap_or_else(|e| panic!("fused layer codegen failed: {e}"));
    let fused_layer_path = out_dir.join("fused_layer.cu");
    std::fs::write(&fused_layer_path, &fused_layer_source)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", fused_layer_path.display()));
    cu_files.push(fused_layer_path.display().to_string());

    // Phase 4 step 4 — scheduled megakernel emission is now DSL-driven.
    // The single source of truth is models/llama.dsl in the macros-core
    // crate; this loop walks the variants block and emits one .cu per
    // declared variant. Adding a new model variant means editing the DSL,
    // not touching this build script.
    let variants =
        vllm_tk_macros_core::generate_scheduled_prefill_variants(vllm_tk_macros_core::LLAMA_DSL)
            .unwrap_or_else(|e| panic!("failed to emit scheduled megakernel variants: {e}"));
    println!(
        "cargo:warning=scheduled megakernel: emitting {} variant(s)",
        variants.len()
    );
    for v in variants {
        let path = out_dir.join(format!("scheduled_prefill_{}.cu", v.name));
        std::fs::write(&path, &v.cu_source)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
        cu_files.push(path.display().to_string());
        println!(
            "cargo:warning=scheduled megakernel: emitted variant `{}`",
            v.name
        );
    }

    // Build all test kernels into one static library
    // CUTLASS headers (optional — if present, the megakernel can include
    // cute/CUTLASS device-side primitives for cuBLAS-quality GEMM phases).
    let cutlass_root = std::path::PathBuf::from(
        std::env::var("CUTLASS_ROOT").unwrap_or_else(|_| "/home/moosevan/cutlass".to_string()),
    );
    let cutlass_include = cutlass_root.join("include");
    let cutlass_tools_util = cutlass_root.join("tools/util/include");

    // FlashInfer headers — pinned via cudaforge git dependency. Cudaforge
    // clones+caches the repo at the pinned commit into
    // `~/.cudaforge/git/checkouts/flashinfer-<hash>/`, shared across
    // worktrees and version-locked by the SHA below. Used by upcoming
    // attention/gemm/norm/rope tile bodies that call FlashInfer device-
    // side primitives instead of being hand-written.
    //
    // Bump this commit deliberately and rerun goldens.
    const FLASHINFER_COMMIT: &str = "08ab45d67705b301ee66e63c6999c934c72dd41c";

    // Vendored FlashInfer instantiation shim — hand-rendered .inc + thin
    // C++ wrapper around BlockBatchPagedAttentionPersistent::Run. Lives in
    // crates/vllm-tk-test-harness/csrc/.
    let harness_csrc = manifest_dir.join("csrc");
    let shim_cu = harness_csrc.join("flashinfer_attention_shim.cu");
    cu_files.push(shim_cu.display().to_string());

    let mut builder = cudaforge::KernelBuilder::new();
    builder = builder
        .out_dir(&cache_dir)
        .source_files(cu_files)
        .watch(header_files)
        .include_path(tk_include.display().to_string())
        .include_path(tk_prototype.display().to_string())
        .include_path(tk_csrc.display().to_string())
        .include_path(harness_csrc.display().to_string())
        .with_git_dependency(
            "flashinfer",
            "https://github.com/flashinfer-ai/flashinfer.git",
            FLASHINFER_COMMIT,
            vec!["include"],
            /*recurse_submodules=*/ false,
        );
    if cutlass_include.exists() {
        println!("cargo:warning=cutlass found at {}", cutlass_root.display());
        builder = builder
            .include_path(cutlass_include.display().to_string())
            .include_path(cutlass_tools_util.display().to_string());
    } else {
        println!(
            "cargo:warning=cutlass not found at {} (set CUTLASS_ROOT)",
            cutlass_root.display()
        );
    }
    builder
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
        .arg("-lineinfo")
        .build_lib(format!("{cache_str}/libtk_test_ops.a"))
        .expect("failed to build test op kernels");

    // Link directives
    println!("cargo:rustc-link-search={cache_str}");
    println!("cargo:rustc-link-lib=static=tk_test_ops");

    // CP4: vllm-rs's fused kernel static lib (built by the
    // sibling vllm-kernels-cuda crate). The harness FFIs the
    // bf16 entry points (`fused_add_rms_norm_bf16`,
    // `silu_and_mul_fused_bf16`, `fused_qkv_rope_cache_bf16`)
    // directly so the natural sm_89 lowering can call vllm-rs's
    // fused kernels as host-callback implementations — same code
    // path as vllm-rs eager.
    let vllm_kernels_dir = std::env::var("HOME")
        .map(|h| format!("{h}/.cache/cudaforge/vllm-cuda"))
        .unwrap_or_else(|_| "/tmp/cudaforge/vllm-cuda".into());
    println!("cargo:rustc-link-search={vllm_kernels_dir}");
    println!("cargo:rustc-link-lib=static=vllm_kernels");

    // CUDA toolkit
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".into());
    println!("cargo:rustc-link-search={cuda_path}/lib64");
    println!("cargo:rustc-link-search={cuda_path}/lib");
    println!("cargo:rustc-link-lib=static=cudart_static");
    // CP4: cuBLAS for the GEMM host-callback implementation.
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=cuda");

    // Rerun triggers
    println!("cargo:rerun-if-changed=build.rs");
    for entry in walkdir(&tk_csrc) {
        let path_str = entry.display().to_string();
        if path_str.ends_with(".cu") || path_str.ends_with(".cuh") {
            println!("cargo:rerun-if-changed={}", path_str);
        }
    }
}

#[cfg(feature = "cuda")]
fn walkdir(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(walkdir(&path));
            } else {
                files.push(path);
            }
        }
    }
    files
}
