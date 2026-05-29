// SPDX-License-Identifier: Apache-2.0
//! Build script for `ferrite-wavefront`.
//!
//! Under the `cuda` feature, emits the linker directives needed to
//! resolve the orchestrator's `launch_<name>` C-linkage host wrappers
//! (declared in `src/launcher.rs`) against the `libmegakernels.a`
//! archive that `ferrite-cuda-builder/build.rs` populates in the
//! cudaforge cache.
//!
//! We can't take a Cargo dep on `ferrite-cuda-builder` from here (it
//! would cycle via `ferrite-models -> ferrite-forward -> us`), so this
//! build.rs just points the linker at the cache and trusts the
//! top-level binary to build both crates in the same workspace
//! invocation.

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let cache = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("cudaforge/vllm-cuda");
    println!("cargo:rustc-link-search=native={}", cache.display());

    // CUDA toolkit (for cudart_static symbols the launcher pulls in
    // via cudaFuncSetAttribute / cudaGetLastError). Honour CUDA_HOME
    // first, then fall through to the standard `/usr/local/cuda`
    // symlink, then pick the highest-versioned `cuda-*` directory.
    let cuda_lib = std::env::var("CUDA_HOME")
        .ok()
        .map(|h| std::path::PathBuf::from(h).join("lib64"))
        .filter(|p| p.exists())
        .or_else(|| {
            let p = std::path::PathBuf::from("/usr/local/cuda/lib64");
            p.exists().then_some(p)
        })
        .or_else(|| {
            let mut cands: Vec<_> = std::fs::read_dir("/usr/local")
                .ok()?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("cuda-"))
                })
                .collect();
            cands.sort();
            cands.into_iter().last().map(|p| p.join("lib64"))
        });
    if let Some(p) = cuda_lib {
        println!("cargo:rustc-link-search=native={}", p.display());
    }

    println!("cargo:rustc-link-lib=static=megakernels");
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!(
        "cargo:rerun-if-changed={}",
        cache.join("libmegakernels.a").display()
    );
}
