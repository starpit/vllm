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

/// Phase 6 single-op fixture: `out = a + b` element-wise. Buffer
/// count: 2 sources + 1 op output = 3. Used by `bin/tk_emit_add` and
/// the `add_kernel_matches_cpu_golden` test.
pub fn add_only_input() -> LoweringInput {
    let h = 2048u32;
    LoweringInput {
        sources: vec![
            SourceShape { rows: 1, cols: h }, // 0  a
            SourceShape { rows: 1, cols: h }, // 1  b
        ],
        ops: vec![OpDesc {
            op: LoweredOp::Add,
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    }
}

/// Phase 6 single-op fixture: `out = silu(gate) * up`. Buffer count:
/// 2 sources + 1 op output = 3. Used by `bin/tk_emit_silu_mul` and
/// the `silu_mul_kernel_matches_cpu_golden` test. Llama-1B
/// intermediate dim = 8192.
pub fn silu_mul_only_input() -> LoweringInput {
    let intermediate = 8192u32;
    LoweringInput {
        sources: vec![
            SourceShape {
                rows: 1,
                cols: intermediate,
            }, // 0  gate
            SourceShape {
                rows: 1,
                cols: intermediate,
            }, // 1  up
        ],
        ops: vec![OpDesc {
            op: LoweredOp::SiluMul,
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    }
}

/// Phase 6 single-op fixture: NeoX RoPE rotate on a single Q row of
/// `[1, num_heads * head_dim]`. Llama-1B Q-side: 32 heads × 64 head_dim
/// = 2048 cols. Buffer count: 3 sources (x, cos, sin) + 1 op output = 4.
pub fn rope_rotate_only_input() -> LoweringInput {
    let head_dim = 64u32;
    let num_heads = 32u32;
    let cols = num_heads * head_dim;
    LoweringInput {
        sources: vec![
            SourceShape { rows: 1, cols },                  // 0  x
            SourceShape { rows: 1, cols: head_dim },        // 1  cos
            SourceShape { rows: 1, cols: head_dim },        // 2  sin
        ],
        ops: vec![OpDesc {
            op: LoweredOp::RopeRotate { head_dim },
            m: 1,
            inputs: vec![
                InputRef::Ext(0),
                InputRef::Ext(1),
                InputRef::Ext(2),
            ],
        }],
        result: 0,
    }
}

/// Phase 6 single-op fixture: M=1 GEMM `y = x @ w^T`, x is `[1, k]`,
/// w is `[n, k]`, y is `[1, n]`. Llama-1B q_proj-shape: k=2048,
/// n=2048 (32 q-heads × 64 head_dim). Buffer count: 2 sources + 1 op
/// output = 3.
pub fn gemm_m1_only_input() -> LoweringInput {
    let k = 2048u32;
    let n = 2048u32;
    LoweringInput {
        sources: vec![
            SourceShape { rows: 1, cols: k }, // 0  x
            SourceShape { rows: n, cols: k }, // 1  w
        ],
        ops: vec![OpDesc {
            op: LoweredOp::Gemm { n, k },
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    }
}

/// Stage 4.A — prefill counterpart of [`one_layer_input`]: same DAG
/// shape as one transformer block, but every op's `m == num_tokens`
/// and the rope/attention ops emit prefill IR variants
/// (`RopeMultiToken` / `ReshapeAndCacheMulti` / `AttnPrefill`)
/// instead of the decode `RopeRotate` / `RopeAppend` / `AttnDecode`
/// triple.
///
/// This is the structural shape the bridge emits for `num_tokens > 1`
/// (see `ferrite_forward_macro::to_wavefront::lower_to_wavefront`).
/// The orchestrator panics on these variants until Stage 4.C lands
/// the per-op lowerings; the test
/// `prefill_one_layer_orchestrator_panics_until_stage_4c` in
/// `tk_orchestrate` pins that contract.
pub fn prefill_one_layer_input(num_tokens: u32) -> LoweringInput {
    assert!(num_tokens >= 1);
    let h = 2048u32;
    let kv = 512u32;
    let i = 8192u32;
    let hd = 64u32;
    let m = num_tokens;
    LoweringInput {
        sources: vec![
            SourceShape { rows: m, cols: h },  // 0  x
            SourceShape { rows: 1, cols: h },  // 1  rms_w0
            SourceShape { rows: h, cols: h },  // 2  q_w
            SourceShape { rows: kv, cols: h }, // 3  k_w
            SourceShape { rows: kv, cols: h }, // 4  v_w
            SourceShape { rows: m, cols: hd }, // 5  cos (per-token)
            SourceShape { rows: m, cols: hd }, // 6  sin (per-token)
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
                m,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            },
            OpDesc {
                op: LoweredOp::Gemm { n: h, k: h },
                m,
                inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
            },
            OpDesc {
                op: LoweredOp::RopeMultiToken { head_dim: hd },
                m,
                inputs: vec![InputRef::Op(1), InputRef::Ext(5), InputRef::Ext(6)],
            },
            OpDesc {
                op: LoweredOp::AttnPrefill {
                    num_q_heads: h / hd,
                    num_kv_heads: kv / hd,
                    head_dim: hd,
                    scale: 0.125,
                },
                m,
                inputs: vec![InputRef::Op(2), InputRef::Ext(7), InputRef::Ext(8)],
            },
            OpDesc {
                op: LoweredOp::Gemm { n: h, k: h },
                m,
                inputs: vec![InputRef::Op(3), InputRef::Ext(9)],
            },
            OpDesc {
                op: LoweredOp::Add,
                m,
                inputs: vec![InputRef::Ext(0), InputRef::Op(4)],
            },
            OpDesc {
                op: LoweredOp::RmsNorm { eps: 1e-5 },
                m,
                inputs: vec![InputRef::Op(5), InputRef::Ext(10)],
            },
            OpDesc {
                op: LoweredOp::Gemm { n: i, k: h },
                m,
                inputs: vec![InputRef::Op(6), InputRef::Ext(11)],
            },
            OpDesc {
                op: LoweredOp::Gemm { n: i, k: h },
                m,
                inputs: vec![InputRef::Op(6), InputRef::Ext(12)],
            },
            OpDesc {
                op: LoweredOp::SiluMul,
                m,
                inputs: vec![InputRef::Op(7), InputRef::Op(8)],
            },
            OpDesc {
                op: LoweredOp::Gemm { n: h, k: i },
                m,
                inputs: vec![InputRef::Op(9), InputRef::Ext(13)],
            },
            OpDesc {
                op: LoweredOp::Add,
                m,
                inputs: vec![InputRef::Ext(0), InputRef::Op(10)],
            },
        ],
        result: 11,
    }
}

/// Stage 4.A — minimal single-op `RopeMultiToken` fixture: rotate
/// num_tokens Q rows. All inputs are external (no upstream ops), so
/// the orchestrator reaches the prefill match arm directly without
/// hitting the upstream-handle-missing path the multi-op fixture
/// trips. Used by `stage_4a_orchestrator_panics_on_prefill_until_4c`.
pub fn rope_multi_only_input(num_tokens: u32) -> LoweringInput {
    let head_dim = 64u32;
    let num_heads = 32u32;
    let cols = num_heads * head_dim;
    LoweringInput {
        sources: vec![
            SourceShape { rows: num_tokens, cols },           // 0  x
            SourceShape { rows: num_tokens, cols: head_dim }, // 1  cos
            SourceShape { rows: num_tokens, cols: head_dim }, // 2  sin
        ],
        ops: vec![OpDesc {
            op: LoweredOp::RopeMultiToken { head_dim },
            m: num_tokens,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1), InputRef::Ext(2)],
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
            | LoweredOp::RopeMultiToken { .. }
            | LoweredOp::ReshapeAndCacheMulti { .. }
            | LoweredOp::Silu
            | LoweredOp::Mul => shape_for(desc.inputs[0], &op_shapes, &input.sources).1,
            LoweredOp::Gemm { n, .. } => n,
            LoweredOp::AttnDecode {
                num_q_heads,
                head_dim,
                ..
            }
            | LoweredOp::AttnPrefill {
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
        // E.12: `const` qualifier dropped from source kernel-arg
        // pointers. Most sources are read-only (weights, prefix
        // activations) but `PrefixK` / `PrefixV` source slots are
        // write-targets for RopeAppend's paged-cache writes
        // (`tma::store_async(buf{idx} + slot * row_bytes, ...)`).
        // Keeping `const` here would make `reinterpret_cast<char*>`
        // an nvcc error ("cannot cast away const"). Reads still
        // typecheck against non-const pointers.
        bufs.push(KernelArg {
            ty: "__nv_bfloat16* __restrict__".into(),
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
    // Runtime decode position — the rope arms read this to compute the
    // per-row cos/sin TMA offset (`__decode_position * row_bytes`).
    // Passed as a host-side u32 (the dispatcher D2H copies
    // `ctx.positions[0]` once per dispatch). Registered whenever any
    // op rotates; AttnDecode alone doesn't need it (its position is
    // implicit in the prefix-K/V cache rows the kernel reads).
    if input
        .ops
        .iter()
        .any(|d| matches!(d.op, LoweredOp::RopeRotate { .. } | LoweredOp::RopeAppend { .. }))
    {
        u32_args.push("__decode_position".into());
    }
    // Runtime decode slot — the new token's absolute paged-cache slot
    // index for the K/V cache writes emitted by RopeAppend's storer.
    // Source: `ctx.slot_mapping[0]` (D2H copy, I64 → u32). Registered
    // whenever any op is RopeAppend; pure RopeRotate (Q-side, no
    // cache write) doesn't need it.
    if input
        .ops
        .iter()
        .any(|d| matches!(d.op, LoweredOp::RopeAppend { .. }))
    {
        u32_args.push("__decode_slot".into());
    }
    KernelArgs { bufs, u32_args }
}
