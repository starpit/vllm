// SPDX-License-Identifier: Apache-2.0
//! Build script for the static megakernel.
//!
//! With `--features cuda`:
//! 1. Generates CUDA source for each model variant from the megakernel DSL
//! 2. Writes them to $OUT_DIR/{name}_static.cu
//! 3. Compiles ALL variants via cudaforge into a single .a
//! 4. Links the resulting .a
//!
//! NL (num_layers) is a runtime parameter — models with different layer counts
//! but the same (HD, ID, HDM, NAH, NKH) share a variant.

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();
}

#[cfg(feature = "cuda")]
fn build_cuda() {
    use std::path::PathBuf;

    // ── Variant table ──
    // Each entry: (name, NL_dummy, HD, ID, HDM, NAH, NKH, VS)
    // NL is a dummy value for compile-time verification only — the kernel accepts
    // num_layers at runtime, so models with different layer counts share a variant.
    //
    // Specialization key: (HD, ID, HDM). These dimensions appear in GL type
    // parameters (shared memory tile shapes) and must be compile-time constants.
    // NAH, NKH, VS are also baked in per variant since they feed into constexpr
    // tile count calculations (optimal_out_block) in the globals struct.
    //
    // To add a new variant: add an entry here AND in the megakernel! variants
    // block in lib.rs. The proc-macro generates FFI declarations and dispatch.
    type Variant = (
        &'static str,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    );
    let variants: &[Variant] = &[
        // Llama 1B (hidden=2048, head_dim=64)
        ("llama_sm89_hd2048_hdm64", 16, 2048, 8192, 64, 32, 8, 128256),
        // NOTE: Llama 3B (HD=3072, NAH=24, NKH=8) has GQA_RATIO=3.
        // The attention_prefill kernel requires GQA_RATIO ∈ {4, 8}.
        // Llama 8B (hidden=4096, head_dim=128)
        (
            "llama_sm89_hd4096_hdm128",
            32,
            4096,
            14336,
            128,
            32,
            8,
            128256,
        ),
        // NOTE: Llama 70B (HD=8192) and 405B (HD=16384) exceed sm89 shared memory
        // budget (99328 bytes). These models require tensor parallelism to reduce
        // per-GPU hidden_dim, or a tiled kernel strategy. Add variants here when
        // the kernel supports larger dims.
    ];

    let dsl_body = r#"
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
    "#;

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Set up cudaforge cache directory
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("cudaforge")
        .join("vllm-tk-static");
    std::fs::create_dir_all(&cache_dir).ok();
    let cache_str = cache_dir.display().to_string();

    // Find TK include paths (relative to workspace root)
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let tk_csrc = workspace_root.join("crates/vllm-tk/csrc");
    let tk_include = tk_csrc.join("include");
    let tk_prototype = tk_csrc.join("prototype");

    // Collect all TK headers for content-hash tracking
    let header_files: Vec<String> = walkdir(&tk_include)
        .iter()
        .chain(walkdir(&tk_csrc).iter())
        .filter(|p| {
            let s = p.display().to_string();
            s.ends_with(".cuh") || s.ends_with(".cu") || s.ends_with(".h")
        })
        .map(|p| p.display().to_string())
        .collect();

    // Generate CUDA source for each variant
    let mut cu_files = Vec::new();
    for &(name, nl, hd, id, hdm, nah, nkh, vs) in variants {
        let dsl = format!(
            "kernel {name}<NL={nl}, HD={hd}, ID={id}, HDM={hdm}, NAH={nah}, NKH={nkh}, VS={vs}> {{\n{dsl_body}\n        }}"
        );

        let cuda_source = vllm_tk_macros_core::generate_cuda_from_dsl(&dsl)
            .unwrap_or_else(|e| panic!("DSL generation failed for variant {name}: {e}"));

        let cu_path = out_dir.join(format!("{name}_static.cu"));
        std::fs::write(&cu_path, &cuda_source)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", cu_path.display()));
        cu_files.push(cu_path.display().to_string());
    }

    // Build ALL variants into one static library
    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(cu_files)
        .watch(header_files)
        .include_path(tk_include.display().to_string())
        .include_path(tk_prototype.display().to_string())
        .include_path(tk_csrc.display().to_string())
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
        .build_lib(format!("{cache_str}/libtk_llama_static.a"))
        .expect("failed to build static megakernel");

    // Link directives
    println!("cargo:rustc-link-search={cache_str}");
    println!("cargo:rustc-link-lib=static=tk_llama_static");

    // CUDA toolkit
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".into());
    println!("cargo:rustc-link-search={cuda_path}/lib64");
    println!("cargo:rustc-link-search={cuda_path}/lib");
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=cuda");

    // Rerun if DSL or TK sources change (including transitive headers like gl.cuh)
    println!("cargo:rerun-if-changed=build.rs");
    for entry in walkdir(&tk_include) {
        println!("cargo:rerun-if-changed={}", entry.display());
    }
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
