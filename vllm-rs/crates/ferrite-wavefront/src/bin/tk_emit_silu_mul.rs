// SPDX-License-Identifier: Apache-2.0
//! Single-op (SiluMul) reproducer for the Phase 6 tile-golden harness.

use std::path::PathBuf;

use ferrite_wavefront::fixtures::{orchestrator_kernel_args, silu_mul_only_input};
use ferrite_wavefront::tk_codegen::{emit_kernel_with_opts, EmitOpts};
use ferrite_wavefront::tk_orchestrate::lower_to_tk;

const KERNEL_NAME: &str = "tk_decode_silu_mul";

fn cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("TK_EMIT_DIR") {
        return PathBuf::from(p);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("cudaforge/megakernels")
}

fn main() {
    let out_dir = cache_dir();
    std::fs::create_dir_all(&out_dir).expect("mkdir cudaforge/megakernels");

    let input = silu_mul_only_input();
    let (prog, n_bufs) = lower_to_tk(&input);
    let args = orchestrator_kernel_args(&input, n_bufs);

    let debug_handshake = std::env::var("TK_EMIT_DEBUG_HANDSHAKE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    let opts = EmitOpts { debug_handshake };
    let src = emit_kernel_with_opts(KERNEL_NAME, &args, &prog, &opts);

    let path = out_dir.join(format!("{KERNEL_NAME}.cu"));
    std::fs::write(&path, &src).expect("write .cu");
    eprintln!(
        "wrote {} ({} bytes, {} bufs, {} ops, debug_handshake={})",
        path.display(),
        src.len(),
        n_bufs,
        input.ops.len(),
        debug_handshake,
    );
}
