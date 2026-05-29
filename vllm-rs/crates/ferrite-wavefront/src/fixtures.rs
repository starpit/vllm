// SPDX-License-Identifier: Apache-2.0
//! Shared `LoweringInput` fixtures used by orchestrator tests, the
//! `dump_tk` example, and the `tk_emit_decode` binary.
//!
//! These are test/dev fixtures only — production wiring derives
//! `LoweringInput` directly from a solved decode FUF via the proc-macro.

use crate::lower::{InputRef, LoweredOp, LoweringInput, OpDesc};
use crate::subtile::SourceShape;

/// Minimal one-layer Llama-3.2-1B-style decode forward.
///
/// Source IDs:
///   0: x         [1, 2048]
///   1: rms_w0    [1, 2048]
///   2: q_w       [2048, 2048]
///   3: k_w       [512, 2048]
///   4: v_w       [512, 2048]
///   5: cos       [1, 64]
///   6: sin       [1, 64]
///   7: k_cache   [1, 2048]      (slice stand-in)
///   8: v_cache   [1, 2048]      (slice stand-in)
///   9: o_w       [2048, 2048]
///  10: rms_w1    [1, 2048]
///  11: gate_w    [8192, 2048]
///  12: up_w      [8192, 2048]
///  13: down_w    [2048, 8192]
pub fn one_layer_input() -> LoweringInput {
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
                // Llama-3.2-1B GQA: 32 q-heads, 8 kv-heads, head_dim=64
                // (q_dim = h = 2048, kv_dim = kv = 512, hd = 64).
                op: LoweredOp::AttnDecode {
                    num_q_heads: h / hd,    // 32
                    num_kv_heads: kv / hd,  // 8
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

/// Single-op RmsNorm reproducer for the orchestrator deadlock audit.
///
/// Layout:
///   Sources: 0 = x [1, hidden], 1 = rms_w [1, hidden]
///   Ops:     0 = RmsNorm(x, rms_w)
///   Result:  op 0
///
/// Buffer count: 2 sources + 1 op output = 3.
///
/// Used by `bin/tk_emit_rmsnorm` and the
/// `launcher_runs_on_zeros_rmsnorm_only` smoke test in
/// [`crate::launcher`]. The fixture isolates the round protocol of one
/// op so a deadlock can be pinned to RmsNorm's specific
/// loader/consumer/storer handshake — versus the 12-op forward, where a
/// bug anywhere in the orchestrator hangs the whole tape.
pub fn rmsnorm_only_input() -> LoweringInput {
    let h = 2048u32;
    LoweringInput {
        sources: vec![
            SourceShape { rows: 1, cols: h }, // 0  x
            SourceShape { rows: 1, cols: h }, // 1  rms_w
        ],
        ops: vec![OpDesc {
            op: LoweredOp::RmsNorm { eps: 1e-5 },
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    }
}

/// Per-buffer byte sizes (bf16 = 2 bytes / element) for a
/// [`LoweringInput`], in `BufId` order: sources first, then per-op
/// output staging buffers. Mirrors the shape inference in
/// `tk_orchestrate::lower_to_tk` so the smoke-test harness can
/// allocate the exact set of device buffers the orchestrator's
/// emitted kernel expects.
pub fn buf_byte_sizes(input: &LoweringInput) -> Vec<usize> {
    let n_sources = input.sources.len();
    let mut out = Vec::with_capacity(n_sources + input.ops.len());

    for s in &input.sources {
        out.push((s.rows as usize) * (s.cols as usize) * 2);
    }

    let shape_for = |r: InputRef,
                     op_shapes: &[(u32, u32)],
                     srcs: &[crate::subtile::SourceShape]|
     -> (u32, u32) {
        match r {
            InputRef::Ext(e) => (srcs[e].rows, srcs[e].cols),
            InputRef::Op(j) => op_shapes[j],
        }
    };

    let mut op_shapes: Vec<(u32, u32)> = Vec::with_capacity(input.ops.len());
    for desc in &input.ops {
        let m = desc.m;
        let cols = match desc.op {
            LoweredOp::RmsNorm { .. }
            | LoweredOp::Add
            | LoweredOp::SiluMul
            | LoweredOp::RopeRotate { .. }
            | LoweredOp::RopeAppend { .. }
            | LoweredOp::Silu
            | LoweredOp::Mul => shape_for(desc.inputs[0], &op_shapes, &input.sources).1,
            LoweredOp::Gemm { n, .. } => n,
            LoweredOp::AttnDecode {
                num_q_heads,
                head_dim,
                ..
            } => num_q_heads * head_dim,
        };
        op_shapes.push((m, cols));
        out.push((m as usize) * (cols as usize) * 2);
    }

    out
}

/// Synthesize the kernel arg signature for an orchestrator output.
///
/// Buffer-id convention (see `tk_orchestrate`):
///   `BufId(0..n_sources)`         — external inputs.
///   `BufId(n_sources..n_sources+n_ops)` — per-op output staging buffers.
///
/// Production wiring will derive types and names from the macro-side
/// `BufferRef` table; this helper just emits one `__nv_bfloat16*` per
/// buffer and a single `__num_kv_pages` runtime arg if any AttnDecode op
/// is present.
pub fn orchestrator_kernel_args(
    input: &LoweringInput,
    n_bufs: u32,
) -> crate::tk_codegen::KernelArgs {
    use crate::tk_codegen::{KernelArg, KernelArgs};
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
