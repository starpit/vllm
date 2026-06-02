// SPDX-License-Identifier: Apache-2.0
//! Single-op (RmsNorm) reproducer for the orchestrator deadlock audit.
//!
//! Emits the orchestrator's lowering of [`fixtures::rmsnorm_only_input`]
//! into the cudaforge megakernel cache as `tk_decode_rmsnorm.cu`, with
//! a parallel C-linkage `launch_tk_decode_rmsnorm` host wrapper. The
//! ferrite-cuda-builder build script picks up every `.cu` in the cache
//! directory, so the symbol joins `tk_decode_one_layer` in the same
//! `libmegakernels.a` archive without disturbing the existing kernel.
//!
//! Run from the workspace root:
//!   cargo run -p ferrite-wavefront --bin tk_emit_rmsnorm
//!
//! Set `TK_EMIT_DEBUG_HANDSHAKE=1` to interleave lane-0-gated `printf`
//! calls around every loader/consumer/storer wait, arrive, and TMA
//! load/store. The first `WAIT_START` line on H100 without a matching
//! prior `ARRIVE` (or a `TMA_LOAD_ISSUED`) is the round-protocol bug
//! the audit is hunting. Override the destination directory by setting
//! `TK_EMIT_DIR` (defaults to `$XDG_CACHE_HOME/cudaforge/megakernels`).

use std::path::PathBuf;

use ferrite_wavefront::fixtures::{orchestrator_kernel_args, rmsnorm_only_input};
use ferrite_wavefront::tk_codegen::{emit_kernel_with_opts, EmitOpts};
use ferrite_wavefront::tk_orchestrate::lower_to_tk;

const KERNEL_NAME: &str = "tk_decode_rmsnorm";

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

    let input = rmsnorm_only_input();
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
