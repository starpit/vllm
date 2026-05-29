// SPDX-License-Identifier: Apache-2.0
//! Dump the rendered .cu for the two vertical slices (RmsNorm and
//! AttnDecode) so they can be eyeballed and fed to nvcc on the pod.
//!
//! Run from the workspace root:
//!   cargo run -p ferrite-wavefront --example dump_tk -- <out_dir>
//!
//! Writes:
//!   <out_dir>/tk_rmsnorm_decode_h2048.cu
//!   <out_dir>/tk_attn_decode_h128.cu

use std::path::PathBuf;

use ferrite_wavefront::subtile_ir::BufId;
use ferrite_wavefront::tk_codegen::{emit_kernel, KernelArg, KernelArgs};
use ferrite_wavefront::tk_lower::{
    lower_attn_decode, lower_rmsnorm, AttnDecodeOp, PageAllocator, RmsNormOp,
};
use ferrite_wavefront::tk_warp_ir::TkProgram;

fn main() {
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/tk_dump"));
    std::fs::create_dir_all(&out_dir).expect("mkdir out_dir");

    // ── RmsNorm ──
    let mut pages = PageAllocator::new();
    let mut prog = TkProgram::new();
    lower_rmsnorm(
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

    // ── AttnDecode ──
    let mut pages2 = PageAllocator::new();
    let mut prog2 = TkProgram::new();
    lower_attn_decode(
        AttnDecodeOp {
            q: BufId(0),
            k_cache: BufId(1),
            v_cache: BufId(2),
            out: BufId(3),
            head_dim: 128,
            act_elem: 2,
            softmax_scale: 0.088_388_35,
            num_kv_pages_arg: "__num_kv_pages",
        },
        &mut pages2,
        &mut prog2,
    );
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
}
