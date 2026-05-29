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

use ferrite_wavefront::lower::{InputRef, LoweredOp, LoweringInput, OpDesc};
use ferrite_wavefront::subtile::SourceShape;
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
    lower_attn_decode::<Phase0>(
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

    // ── Full decode: one-layer forward via the orchestrator ──
    //
    // Source shapes match a Llama-3.2-1B-style decode (test fixture only
    // — production pipeline picks shapes from `LoweringInput.sources`).
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

/// Synthesize the kernel arg signature for an orchestrator output.
///
/// Buffer-id convention (see `tk_orchestrate`):
///   `BufId(0..n_sources)`         — external inputs (model weights /
///                                   activations / KV slices).
///   `BufId(n_sources..n_sources+n_ops)` — per-op output staging buffers.
///
/// Production wiring will derive types and names from the macro-side
/// `BufferRef` table; the example just emits one `__nv_bfloat16*` per
/// buffer and a single `__num_kv_pages` runtime arg (used by every
/// AttnDecode op).
fn orchestrator_kernel_args(input: &LoweringInput, n_bufs: u32) -> KernelArgs {
    let n_sources = input.sources.len() as u32;
    let mut bufs = Vec::with_capacity(n_bufs as usize);
    for i in 0..n_sources {
        bufs.push(KernelArg {
            ty: "const __nv_bfloat16* __restrict__".into(),
            name: format!("src{i}"),
        });
    }
    for i in 0..(n_bufs - n_sources) {
        bufs.push(KernelArg {
            ty: "__nv_bfloat16* __restrict__".into(),
            name: format!("op{i}_out"),
        });
    }
    let mut u32_args = vec![];
    if input
        .ops
        .iter()
        .any(|d| matches!(d.op, LoweredOp::AttnDecode { .. }))
    {
        u32_args.push("__num_kv_pages".into());
    }
    KernelArgs { bufs, u32_args }
}

/// Minimal one-layer decode forward (Llama-3.2-1B-style). Test fixture
/// only — the production pipeline (proc-macro → `lower_decode_to_wavefront`)
/// produces a `LoweringInput` directly from the solved decode FUF.
fn one_layer_input() -> LoweringInput {
    let h = 2048u32;
    let kv = 512u32;
    let i = 8192u32;
    let hd = 64u32;
    LoweringInput {
        sources: vec![
            SourceShape { rows: 1, cols: h },  // 0  x
            SourceShape { rows: 1, cols: h },  // 1  rms_w0
            SourceShape { rows: h, cols: h },  // 2  q_w
            SourceShape { rows: kv, cols: h }, // 3  k_w
            SourceShape { rows: kv, cols: h }, // 4  v_w
            SourceShape { rows: 1, cols: hd }, // 5  cos
            SourceShape { rows: 1, cols: hd }, // 6  sin
            SourceShape { rows: 1, cols: h },  // 7  k_cache slice
            SourceShape { rows: 1, cols: h },  // 8  v_cache slice
            SourceShape { rows: h, cols: h },  // 9  o_w
            SourceShape { rows: 1, cols: h },  // 10 rms_w1
            SourceShape { rows: i, cols: h },  // 11 gate_w
            SourceShape { rows: i, cols: h },  // 12 up_w
            SourceShape { rows: h, cols: i },  // 13 down_w
        ],
        ops: vec![
            OpDesc {
                op: LoweredOp::RmsNorm { eps: 1e-5 },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            },
            OpDesc {
                op: LoweredOp::Gemm { n: h, k: h },
                m: 1,
                inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
            },
            OpDesc {
                op: LoweredOp::RopeRotate { head_dim: hd },
                m: 1,
                inputs: vec![InputRef::Op(1), InputRef::Ext(5), InputRef::Ext(6)],
            },
            OpDesc {
                op: LoweredOp::AttnDecode {
                    num_q_heads: 1,
                    num_kv_heads: 1,
                    head_dim: hd,
                    scale: 0.125,
                },
                m: 1,
                inputs: vec![InputRef::Op(2), InputRef::Ext(7), InputRef::Ext(8)],
            },
            OpDesc {
                op: LoweredOp::Gemm { n: h, k: h },
                m: 1,
                inputs: vec![InputRef::Op(3), InputRef::Ext(9)],
            },
            OpDesc {
                op: LoweredOp::Add,
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Op(4)],
            },
            OpDesc {
                op: LoweredOp::RmsNorm { eps: 1e-5 },
                m: 1,
                inputs: vec![InputRef::Op(5), InputRef::Ext(10)],
            },
            OpDesc {
                op: LoweredOp::Gemm { n: i, k: h },
                m: 1,
                inputs: vec![InputRef::Op(6), InputRef::Ext(11)],
            },
            OpDesc {
                op: LoweredOp::Gemm { n: i, k: h },
                m: 1,
                inputs: vec![InputRef::Op(6), InputRef::Ext(12)],
            },
            OpDesc {
                op: LoweredOp::SiluMul,
                m: 1,
                inputs: vec![InputRef::Op(7), InputRef::Op(8)],
            },
            OpDesc {
                op: LoweredOp::Gemm { n: h, k: i },
                m: 1,
                inputs: vec![InputRef::Op(9), InputRef::Ext(13)],
            },
            OpDesc {
                op: LoweredOp::Add,
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Op(10)],
            },
        ],
        result: 11,
    }
}
