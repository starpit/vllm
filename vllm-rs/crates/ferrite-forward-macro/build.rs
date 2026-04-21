// SPDX-License-Identifier: Apache-2.0
// Propagate `FERRITE_DISABLE_CUBLAS_GEMM` into rustc's cmdline as a
// `--cfg` flag. The `rerun-if-env-changed` directive makes cargo
// re-invoke build.rs when the var changes; emitting a different cfg
// changes the rustc cmdline, which forces `ferrite-forward-macro`
// itself to rebuild — which cascades to every downstream crate that
// expands `forward!`. A plain `env::var()` read at proc-macro
// expansion time doesn't reach cargo's invalidation graph.
fn main() {
    println!("cargo:rerun-if-env-changed=FERRITE_DISABLE_CUBLAS_GEMM");
    if std::env::var_os("FERRITE_DISABLE_CUBLAS_GEMM").is_some() {
        println!("cargo:rustc-cfg=disable_cublas_gemm");
    }
    // Keep the unused-cfg lint quiet when the flag is not emitted.
    println!("cargo:rustc-check-cfg=cfg(disable_cublas_gemm)");
}
