use std::path::PathBuf;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let target_dir = PathBuf::from(&out_dir)
        .ancestors()
        .find(|p| p.ends_with("build"))
        .unwrap()
        .to_path_buf();

    // Find mlx-sys build directory with MLX headers.
    let mlx_sys_dir = std::fs::read_dir(&target_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| {
            let n = e.file_name();
            let n = n.to_str().unwrap_or("");
            n.starts_with("mlx-sys-") && e.path().join("out/build/_deps/mlx-src").exists()
        })
        .map(|e| e.path().join("out/build/_deps"));

    let mlx_sys_dir = match mlx_sys_dir {
        Some(d) => d,
        None => {
            eprintln!("cargo:warning=mlx-sys build dir not found, skipping custom kernels");
            return;
        }
    };

    let mlx_src = mlx_sys_dir.join("mlx-src");
    let metal_cpp = mlx_sys_dir.join("metal_cpp-src");

    let mlx_c_src = PathBuf::from(std::env::var("DEP_MLX_C_INCLUDE").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!(
            "{home}/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/mlx-sys-0.2.0/src/mlx-c"
        )
    }));

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(2)
        .warnings(false) // suppress upstream MLX header warnings
        .include(&mlx_src)
        .include(&metal_cpp)
        .include(&mlx_c_src)
        .include("csrc")
        .file("csrc/multi_segment_sdpa.cpp")
        .compile("vllm_mlx_kernels");

    println!("cargo:rerun-if-changed=csrc/multi_segment_sdpa.cpp");
    println!("cargo:rerun-if-changed=csrc/multi_segment_sdpa.h");
}
