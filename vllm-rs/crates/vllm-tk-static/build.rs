// SPDX-License-Identifier: Apache-2.0
//! Build script for the static megakernel.
//!
//! With `--features cuda`:
//! 1. Generates CUDA source from the megakernel DSL (via vllm-tk-macros-core)
//! 2. Writes it to $OUT_DIR/llama_sm89_static.cu
//! 3. Compiles via cudaforge with the same flags as vllm-tk
//! 4. Links the resulting .a

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();
}

#[cfg(feature = "cuda")]
fn build_cuda() {
    use std::path::PathBuf;

    // Generate the CUDA source using the same DSL the proc-macro parses
    let dsl = r#"
        kernel llama_sm89<NL=16, HD=2048, ID=8192, HDM=64, NAH=32, NKH=8, VS=128256> {
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
        }
    "#;

    let cuda_source = vllm_tk_macros_core::generate_cuda_from_dsl(dsl)
        .expect("megakernel DSL generation failed in build.rs");

    // Write generated CUDA to OUT_DIR
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cu_path = out_dir.join("llama_sm89_static.cu");
    std::fs::write(&cu_path, &cuda_source).expect("failed to write generated .cu");

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

    // TK headers are in the vllm-tk crate's csrc/
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

    // Build with cudaforge
    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(vec![cu_path.display().to_string()])
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
