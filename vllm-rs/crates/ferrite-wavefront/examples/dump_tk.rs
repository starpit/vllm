// SPDX-License-Identifier: Apache-2.0
//! Dump rendered .cu sources for the TK 2.0 typed-warp-tier substrate so
//! they can be eyeballed and fed to nvcc on the pod.
//!
//! Run from the workspace root:
//!   cargo run -p ferrite-wavefront --example dump_tk -- <out_dir>
//!
//! Writes:
//!   <out_dir>/tk_rmsnorm_decode_h2048.cu       — single RmsNorm slice
//!   <out_dir>/tk_attn_decode_h128.cu           — single AttnDecode slice
//!   <out_dir>/tk_decode_one_layer.cu           — full one-layer decode
//!                                                forward via the
//!                                                orchestrator (the
//!                                                end-to-end megakernel
//!                                                target)

use std::path::PathBuf;

use ferrite_wavefront::fixtures::{one_layer_input, orchestrator_kernel_args};
use ferrite_wavefront::subtile_ir::BufId;
use ferrite_wavefront::tk_codegen::{emit_kernel, KernelArg, KernelArgs};
use ferrite_wavefront::tk_lower::{
    lower_attn_decode, lower_rmsnorm, AttnDecodeOp, PageAllocator, RmsNormOp,
};
use ferrite_wavefront::tk_orchestrate::lower_to_tk;
use ferrite_wavefront::tk_warp_ir::{Phase0, TkProgram};

fn main() {
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/tk_dump"));
    std::fs::create_dir_all(&out_dir).expect("mkdir out_dir");

    // ── Slice: RmsNorm ──
    let mut pages = PageAllocator::new();
    let mut prog = TkProgram::new();
    lower_rmsnorm::<Phase0>(
        RmsNormOp {
            x: BufId(0),
            weight: BufId(1),
            out: BufId(2),
            hidden: 2048,
            m: 1,
            act_elem: 2,
            eps: 1e-5,
            init: true,
        },
        &mut pages,
        &mut prog,
    );
    let args = KernelArgs {
        bufs: vec![
            KernelArg {
                ty: "const __nv_bfloat16* __restrict__".into(),
                name: "x".into(),
            },
            KernelArg {
                ty: "const __nv_bfloat16* __restrict__".into(),
                name: "weight".into(),
            },
            KernelArg {
                ty: "__nv_bfloat16* __restrict__".into(),
                name: "out".into(),
            },
        ],
        u32_args: vec![],
    };
    let src = emit_kernel("tk_rmsnorm_decode_h2048", &args, &prog);
    let path = out_dir.join("tk_rmsnorm_decode_h2048.cu");
    std::fs::write(&path, &src).unwrap();
    eprintln!("wrote {} ({} bytes)", path.display(), src.len());

    // ── Slice: AttnDecode ──
    let mut pages2 = PageAllocator::new();
    let mut prog2 = TkProgram::new();
    let attn_op = AttnDecodeOp {
        q: BufId(0),
        k_cache: BufId(1),
        v_cache: BufId(2),
        out: BufId(3),
        head_dim: 128,
        num_q_heads: 8,
        num_kv_heads: 8,
        act_elem: 2,
        softmax_scale: 0.088_388_35,
        num_kv_pages_arg: ferrite_wavefront::tk_warp_ir::NumKvPagesSym,
        unique_id: 0,
    };
    let k = ferrite_wavefront::tk_gmem::GmemHandle::<ferrite_wavefront::tk_gmem::KCache>::new_initial(
        attn_op.k_cache,
    );
    let v = ferrite_wavefront::tk_gmem::GmemHandle::<ferrite_wavefront::tk_gmem::VCache>::new_initial(
        attn_op.v_cache,
    );
    let kf = ferrite_wavefront::tk_gmem::emit_fence_after_op(&mut prog2, k);
    let vf = ferrite_wavefront::tk_gmem::emit_fence_after_op(&mut prog2, v);
    lower_attn_decode::<Phase0>(attn_op, kf, vf, &mut pages2, &mut prog2);
    let args2 = KernelArgs {
        bufs: vec![
            KernelArg {
                ty: "const __nv_bfloat16* __restrict__".into(),
                name: "q".into(),
            },
            KernelArg {
                ty: "const __nv_bfloat16* __restrict__".into(),
                name: "k_cache".into(),
            },
            KernelArg {
                ty: "const __nv_bfloat16* __restrict__".into(),
                name: "v_cache".into(),
            },
            KernelArg {
                ty: "__nv_bfloat16* __restrict__".into(),
                name: "out".into(),
            },
        ],
        u32_args: vec!["__num_kv_pages".into()],
    };
    let src2 = emit_kernel("tk_attn_decode_h128", &args2, &prog2);
    let path2 = out_dir.join("tk_attn_decode_h128.cu");
    std::fs::write(&path2, &src2).unwrap();
    eprintln!("wrote {} ({} bytes)", path2.display(), src2.len());

    // ── Full decode: one-layer forward via the orchestrator ──
    let input = one_layer_input();
    let (prog3, n_bufs) = lower_to_tk(&input);
    let args3 = orchestrator_kernel_args(&input, n_bufs);
    let src3 = emit_kernel("tk_decode_one_layer", &args3, &prog3);
    let path3 = out_dir.join("tk_decode_one_layer.cu");
    std::fs::write(&path3, &src3).unwrap();
    eprintln!(
        "wrote {} ({} bytes, {} bufs, {} ops)",
        path3.display(),
        src3.len(),
        n_bufs,
        input.ops.len()
    );
}
