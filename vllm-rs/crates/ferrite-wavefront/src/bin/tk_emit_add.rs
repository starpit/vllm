// SPDX-License-Identifier: Apache-2.0
//! Single-op (Add) reproducer for the Phase 6 tile-golden harness.
//!
//! Emits the orchestrator's lowering of [`fixtures::add_only_input`]
//! into the cudaforge megakernel cache as `tk_decode_add.cu`, with a
//! parallel C-linkage `launch_tk_decode_add` host wrapper. Picked up
//! by `ferrite-cuda-builder`'s build script alongside every other
//! `.cu` in the cache.
//!
//! Run from workspace root:
//!   cargo run -p ferrite-wavefront --bin tk_emit_add
//!
//! Used by `add_kernel_matches_cpu_golden` test in
//! `crates/ferrite-wavefront/src/launcher.rs`.

use std::path::PathBuf;

use ferrite_wavefront::fixtures::{add_only_input, orchestrator_kernel_args};
use ferrite_wavefront::tk_codegen::{emit_kernel_with_opts, EmitOpts};
use ferrite_wavefront::tk_orchestrate::lower_to_tk;

const KERNEL_NAME: &str = "tk_decode_add";

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

    let input = add_only_input();
    let (prog, n_bufs) = lower_to_tk(&input);
    let args = orchestrator_kernel_args(&input, n_bufs);

    let debug_handshake = std::env::var("TK_EMIT_DEBUG_HANDSHAKE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    let opts = EmitOpts {
        debug_handshake,
        ..Default::default()
    };
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
