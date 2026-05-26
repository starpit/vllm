// SPDX-License-Identifier: Apache-2.0
//! `LoweringInput` → `SubtileGraph`: stitch per-op subtile decompositions
//! into a whole forward.
//!
//! [`LoweringInput`] is a flat, topologically-ordered, backend-agnostic
//! description of a solved forward: one [`OpDesc`] per op, each naming its
//! kind, row count, and inputs (either a prior op's output or an external
//! source — a weight/activation/extern). It is deliberately a *plain data*
//! mirror of the macro crate's internal `Fuf` + solver `Assignment`: the
//! macro crate (which alone can see those types) does the trivial
//! structural translation and calls [`lower`] here, so all the
//! decomposition logic lives in this Mac-testable crate.
//!
//! This first cut lowers at **coarse granularity** — one subtile per op,
//! every consumer reading whole producer outputs via [`Operand::Sub`].
//! That is bit-exact and exercises the inter-op stitching; the per-op
//! tiling (split-K, N-blocks) and the inter-op slicing it needs are layered
//! on with the scheduler, which is where the fine granularity earns its
//! keep.

use crate::subtile::{
    EwKind, Operand, OutputSlot, Range, Region, SourceShape, SubOp, SubtileGraph, SubtileId,
    SubtileNode,
};

/// One input edge of an op: a prior op's output, or an external source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputRef {
    /// Output of op `ops[idx]` (must be `< this op's index`).
    Op(usize),
    /// External source `sources[idx]` — a weight, activation, or extern.
    Ext(usize),
}

/// The op a node performs, with the shape parameters the decomposition
/// needs. Grows alongside [`SubOp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoweredOp {
    /// `out[M, n] = act[M, k] @ W[n, k]^T`. Inputs: `[act, weight]`.
    Gemm { n: u32, k: u32 },
    /// `out[M, d] = rmsnorm(x, weight, eps)`. Inputs: `[x, weight]`.
    RmsNorm { eps: f32 },
    /// `silu(x)`. Input: `[x]`. Shape-preserving.
    Silu,
    /// `a * b`. Inputs: `[a, b]`. Shape-preserving.
    Mul,
    /// Fused `silu(gate) * up`. Inputs: `[gate, up]`. Shape-preserving.
    /// Produced by [`fuse_silu_mul`] from an adjacent `Silu` + `Mul`; maps
    /// to the GPU's fused `silu_mul` arm.
    SiluMul,
    /// `a + b` (e.g. the residual). Inputs: `[a, b]`. Shape-preserving.
    Add,
    /// NeoX rotary over `[M, heads * head_dim]`. Inputs: `[x, cos, sin]`,
    /// where cos/sin are the new token's position rows `[1, head_dim]`.
    /// Shape-preserving.
    RopeRotate { head_dim: u32 },
    /// Fused decode attention. Inputs: `[Q, (K_seg, V_seg)...]` —
    /// concatenated along the KV axis. The fused decode passes the prefix
    /// cache as `Ext` (`Source`) segments and the rotated new token as
    /// `Op` (`Sub`) segments, so the new K/V is an internal dataflow edge,
    /// never a cache round-trip. Output: `[M, num_q_heads * head_dim]`.
    AttnDecode {
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: f32,
    },
}

/// One op in the forward.
#[derive(Clone, Debug)]
pub struct OpDesc {
    pub op: LoweredOp,
    /// Row count (M); decode is 1.
    pub m: u32,
    pub inputs: Vec<InputRef>,
}

/// A whole forward, ready to lower.
#[derive(Clone, Debug)]
pub struct LoweringInput {
    /// External tensors (weights / activations / externs), indexed by
    /// `InputRef::Ext` and by `SourceId` in the produced graph.
    pub sources: Vec<SourceShape>,
    /// Ops in topological order; op `i` may reference ops `< i`.
    pub ops: Vec<OpDesc>,
    /// Which op's output is the forward's result.
    pub result: usize,
}

fn resolve_input(r: InputRef, op_out: &[SubtileId], sources: &[SourceShape]) -> Operand {
    match r {
        InputRef::Op(j) => Operand::Sub(op_out[j]),
        InputRef::Ext(e) => Operand::Source {
            id: crate::subtile::SourceId(e as u32),
            region: Region {
                rows: Range::new(0, sources[e].rows),
                cols: Range::new(0, sources[e].cols),
            },
        },
    }
}

fn input_shape(r: InputRef, op_shape: &[(u32, u32)], sources: &[SourceShape]) -> (u32, u32) {
    match r {
        InputRef::Op(j) => op_shape[j],
        InputRef::Ext(e) => (sources[e].rows, sources[e].cols),
    }
}

/// Lower a whole forward to a coarse subtile DAG (one subtile per op).
pub fn lower(input: &LoweringInput) -> SubtileGraph {
    let mut nodes: Vec<SubtileNode> = Vec::with_capacity(input.ops.len());
    let mut op_out: Vec<SubtileId> = Vec::with_capacity(input.ops.len());
    let mut op_shape: Vec<(u32, u32)> = Vec::with_capacity(input.ops.len());

    for desc in &input.ops {
        let ins: Vec<Operand> = desc
            .inputs
            .iter()
            .map(|r| resolve_input(*r, &op_out, &input.sources))
            .collect();
        let (subop, out_cols) = match desc.op {
            LoweredOp::Gemm { n, k } => {
                let act_cols = input_shape(desc.inputs[0], &op_shape, &input.sources).1;
                assert_eq!(act_cols, k, "gemm activation cols must equal k");
                (SubOp::MatmulTile, n)
            }
            LoweredOp::RmsNorm { eps } => {
                let d = input_shape(desc.inputs[0], &op_shape, &input.sources).1;
                (SubOp::RmsNorm { eps }, d)
            }
            LoweredOp::Silu => {
                let d = input_shape(desc.inputs[0], &op_shape, &input.sources).1;
                (SubOp::Elementwise(EwKind::Silu), d)
            }
            LoweredOp::Mul => {
                let d = input_shape(desc.inputs[0], &op_shape, &input.sources).1;
                (SubOp::Elementwise(EwKind::Mul), d)
            }
            LoweredOp::SiluMul => {
                let d = input_shape(desc.inputs[0], &op_shape, &input.sources).1;
                (SubOp::SiluMul, d)
            }
            LoweredOp::Add => {
                let d = input_shape(desc.inputs[0], &op_shape, &input.sources).1;
                (SubOp::Elementwise(EwKind::Add), d)
            }
            LoweredOp::RopeRotate { head_dim } => {
                let cols = input_shape(desc.inputs[0], &op_shape, &input.sources).1;
                (SubOp::RopeRotate { head_dim }, cols)
            }
            LoweredOp::AttnDecode {
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
            } => (
                SubOp::AttnDecode {
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                },
                num_q_heads * head_dim,
            ),
        };
        let id = SubtileId(nodes.len() as u32);
        nodes.push(SubtileNode {
            id,
            op: subop,
            inputs: ins,
            out_rows: desc.m,
            out_cols,
        });
        op_out.push(id);
        op_shape.push((desc.m, out_cols));
    }

    let (rr, rc) = op_shape[input.result];
    SubtileGraph {
        nodes,
        sources: input.sources.clone(),
        result_rows: rr,
        result_cols: rc,
        outputs: vec![OutputSlot {
            node: op_out[input.result],
            dest: Region {
                rows: Range::new(0, rr),
                cols: Range::new(0, rc),
            },
        }],
    }
}

// ── Silu + Mul → SiluMul fusion ─────────────────────────────────────

/// Fuse each adjacent `Silu` → `Mul(silu, up)` into one
/// [`LoweredOp::SiluMul`] (SwiGLU). The GPU has only a *fused* `silu_mul`
/// arm (no standalone silu), and fusion must happen **before scheduling**
/// so the pair lands on one worker — the serializer otherwise rejects a
/// standalone `Silu`. A `Silu` is fused only when its output feeds exactly
/// one consumer (that `Mul`) and is not the forward result; everything else
/// is left untouched. Bit-exact: `SiluMul` computes `silu(gate) * up`,
/// identical to the two separate ops, so `eval_dag` is unchanged.
///
/// `Mul` is commutative, so the `Silu`-producing operand becomes `gate` and
/// the other becomes `up` regardless of input order. Op references are
/// re-indexed after the dropped `Silu`s are removed.
pub fn fuse_silu_mul(input: &LoweringInput) -> LoweringInput {
    use std::collections::HashMap;
    let n = input.ops.len();

    // Count consumers of each op output (op inputs across the forward + the
    // result) so a multiply-consumed `Silu` is never duplicated by fusion.
    let mut uses = vec![0u32; n];
    for od in &input.ops {
        for r in &od.inputs {
            if let InputRef::Op(j) = r {
                uses[*j] += 1;
            }
        }
    }
    uses[input.result] += 1;

    // Decide fusions: mul index → (silu index, gate ref, up ref).
    let is_silu = |i: usize| matches!(input.ops[i].op, LoweredOp::Silu);
    let mut fuse_at: HashMap<usize, (usize, InputRef, InputRef)> = HashMap::new();
    let mut dropped = vec![false; n];
    for (j, od) in input.ops.iter().enumerate() {
        if !matches!(od.op, LoweredOp::Mul) || od.inputs.len() != 2 {
            continue;
        }
        // A fusable operand is a single-use, non-result `Silu` output.
        let fusable = |r: InputRef, dropped: &[bool]| -> Option<usize> {
            if let InputRef::Op(i) = r
                && is_silu(i)
                && uses[i] == 1
                && input.result != i
                && !dropped[i]
            {
                return Some(i);
            }
            None
        };
        let (a, b) = (od.inputs[0], od.inputs[1]);
        let pick = fusable(a, &dropped)
            .map(|si| (si, b))
            .or_else(|| fusable(b, &dropped).map(|si| (si, a)));
        if let Some((si, up)) = pick {
            let gate = input.ops[si].inputs[0];
            fuse_at.insert(j, (si, gate, up));
            dropped[si] = true;
        }
    }
    if fuse_at.is_empty() {
        return input.clone();
    }

    // Old op index → new index, with the dropped `Silu`s removed.
    let mut new_idx = vec![usize::MAX; n];
    let mut next = 0usize;
    for (i, d) in dropped.iter().enumerate() {
        if !d {
            new_idx[i] = next;
            next += 1;
        }
    }
    let remap = |r: InputRef| -> InputRef {
        match r {
            InputRef::Op(j) => {
                debug_assert_ne!(
                    new_idx[j],
                    usize::MAX,
                    "ref to a dropped Silu survived fusion"
                );
                InputRef::Op(new_idx[j])
            }
            InputRef::Ext(e) => InputRef::Ext(e),
        }
    };

    let mut ops: Vec<OpDesc> = Vec::with_capacity(next);
    for (i, od) in input.ops.iter().enumerate() {
        if dropped[i] {
            continue;
        }
        if let Some((_, gate, up)) = fuse_at.get(&i) {
            ops.push(OpDesc {
                op: LoweredOp::SiluMul,
                m: od.m,
                inputs: vec![remap(*gate), remap(*up)],
            });
        } else {
            ops.push(OpDesc {
                op: od.op,
                m: od.m,
                inputs: od.inputs.iter().map(|r| remap(*r)).collect(),
            });
        }
    }
    LoweringInput {
        sources: input.sources.clone(),
        ops,
        result: new_idx[input.result],
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subtile::{assemble_result, eval_dag};
    use ferrite_forward::cpu_golden;

    fn rng_fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (s >> 33) as u32;
                (bits as f32 / 2147483648.0) * 2.0 - 1.0
            })
            .collect()
    }

    /// rmsnorm(x, wn) → gemm(_, W): proves a producer op's output stitches
    /// into a consumer op via a `Sub` edge, bit-exact end-to-end.
    #[test]
    fn rmsnorm_then_gemm_chain_bit_exact() {
        let (k, n) = (64u32, 48u32);
        let eps = 1e-5f32;
        let x = rng_fill(k as usize, 1);
        let wn = rng_fill(k as usize, 2);
        let w = rng_fill((n * k) as usize, 3);

        // Reference: rmsnorm then gemm.
        let mut xn = vec![0f32; k as usize];
        cpu_golden::rmsnorm(&x, &wn, &mut xn, eps);
        let mut want = vec![0f32; n as usize];
        cpu_golden::gemm(&xn, &w, &mut want, 1, k as usize, n as usize);

        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: k }, // 0: x
                SourceShape { rows: 1, cols: k }, // 1: rmsnorm weight
                SourceShape { rows: n, cols: k }, // 2: gemm weight
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::RmsNorm { eps },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n, k },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
                },
            ],
            result: 1,
        };

        let g = lower(&input);
        // Two ops → two nodes; gemm reads the rmsnorm node via a Sub edge.
        assert_eq!(g.nodes.len(), 2);
        assert!(matches!(g.nodes[1].inputs[0], Operand::Sub(SubtileId(0))));

        let got = assemble_result(&g, &eval_dag(&g, &[&x, &wn, &w]));
        assert_eq!(got, want, "rmsnorm→gemm chain must be bit-exact");
    }

    /// A whole SwiGLU MLP block, decode (M=1): rmsnorm → gate/up gemms →
    /// silu·mul → down gemm → residual add. Lowered via `lower()` and
    /// bit-exact vs the cpu_golden composition. Exercises a diamond
    /// (gate+up both read the norm; mul rejoins them) and the residual
    /// add reading the block's original input.
    #[test]
    fn swiglu_mlp_block_bit_exact() {
        let (h, i) = (32u32, 80u32); // hidden, intermediate
        let eps = 1e-5f32;
        let x = rng_fill(h as usize, 11);
        let norm_w = rng_fill(h as usize, 12);
        let w_gate = rng_fill((i * h) as usize, 13);
        let w_up = rng_fill((i * h) as usize, 14);
        let w_down = rng_fill((h * i) as usize, 15);

        // Reference.
        let mut xn = vec![0f32; h as usize];
        cpu_golden::rmsnorm(&x, &norm_w, &mut xn, eps);
        let mut gate = vec![0f32; i as usize];
        cpu_golden::gemm(&xn, &w_gate, &mut gate, 1, h as usize, i as usize);
        let mut up = vec![0f32; i as usize];
        cpu_golden::gemm(&xn, &w_up, &mut up, 1, h as usize, i as usize);
        let mut act = vec![0f32; i as usize];
        cpu_golden::fused_gate_up_silu_mul(&gate, &up, &mut act);
        let mut down = vec![0f32; h as usize];
        cpu_golden::gemm(&act, &w_down, &mut down, 1, i as usize, h as usize);
        let mut want = vec![0f32; h as usize];
        cpu_golden::add(&down, &x, &mut want);

        // Sources: 0=x, 1=norm_w, 2=W_gate, 3=W_up, 4=W_down.
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: h },
                SourceShape { rows: 1, cols: h },
                SourceShape { rows: i, cols: h },
                SourceShape { rows: i, cols: h },
                SourceShape { rows: h, cols: i },
            ],
            ops: vec![
                // 0: x_norm = rmsnorm(x, norm_w)
                OpDesc {
                    op: LoweredOp::RmsNorm { eps },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                // 1: gate = gemm(x_norm, W_gate)
                OpDesc {
                    op: LoweredOp::Gemm { n: i, k: h },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
                },
                // 2: up = gemm(x_norm, W_up)
                OpDesc {
                    op: LoweredOp::Gemm { n: i, k: h },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(3)],
                },
                // 3: silu(gate)
                OpDesc {
                    op: LoweredOp::Silu,
                    m: 1,
                    inputs: vec![InputRef::Op(1)],
                },
                // 4: silu(gate) * up
                OpDesc {
                    op: LoweredOp::Mul,
                    m: 1,
                    inputs: vec![InputRef::Op(3), InputRef::Op(2)],
                },
                // 5: down = gemm(act, W_down)
                OpDesc {
                    op: LoweredOp::Gemm { n: h, k: i },
                    m: 1,
                    inputs: vec![InputRef::Op(4), InputRef::Ext(4)],
                },
                // 6: residual + down
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(5), InputRef::Ext(0)],
                },
            ],
            result: 6,
        };

        let g = lower(&input);
        let srcs: Vec<&[f32]> = vec![&x, &norm_w, &w_gate, &w_up, &w_down];
        let got = assemble_result(&g, &eval_dag(&g, &srcs));
        assert_eq!(got, want, "SwiGLU MLP block must be bit-exact");

        // Fusing Silu+Mul → SiluMul drops the standalone silu, rewrites the
        // mul, re-indexes the refs, and is bit-exact (same arithmetic).
        let fused = fuse_silu_mul(&input);
        assert_eq!(fused.ops.len(), input.ops.len() - 1, "one op fewer");
        assert_eq!(
            fused
                .ops
                .iter()
                .filter(|o| matches!(o.op, LoweredOp::SiluMul))
                .count(),
            1,
            "exactly one fused SiluMul"
        );
        assert!(
            !fused
                .ops
                .iter()
                .any(|o| matches!(o.op, LoweredOp::Silu | LoweredOp::Mul)),
            "no standalone Silu/Mul remain"
        );
        // The SiluMul reads (gate gemm, up gemm); the down gemm + add are
        // re-pointed; the result is remapped.
        let sm = fused
            .ops
            .iter()
            .find(|o| matches!(o.op, LoweredOp::SiluMul))
            .unwrap();
        assert_eq!(
            sm.inputs,
            vec![InputRef::Op(1), InputRef::Op(2)],
            "gate, up"
        );
        let gf = lower(&fused);
        let got_f = assemble_result(&gf, &eval_dag(&gf, &srcs));
        assert_eq!(got_f, want, "fused SwiGLU MLP must stay bit-exact");
    }

    /// A `Silu` whose output has a second consumer (besides the `Mul`) is
    /// NOT fused (fusion would have to duplicate it); the graph is returned
    /// untouched.
    #[test]
    fn fuse_silu_mul_skips_multi_use_silu() {
        let h = 8u32;
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: h },
                SourceShape { rows: 1, cols: h },
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::Silu,
                    m: 1,
                    inputs: vec![InputRef::Ext(0)],
                },
                OpDesc {
                    op: LoweredOp::Mul,
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(1)],
                },
                // second consumer of the silu output → not single-use
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Op(1)],
                },
            ],
            result: 2,
        };
        let fused = fuse_silu_mul(&input);
        assert_eq!(fused.ops.len(), input.ops.len(), "nothing fused");
        assert!(fused.ops.iter().any(|o| matches!(o.op, LoweredOp::Silu)));
    }

    /// A whole Llama-style decode layer (M=1): input-norm → q/k/v gemms →
    /// rope(q), rope(k) → fused GQA attention over [prefix ++ new] →
    /// o-proj → residual → post-norm → SwiGLU MLP → residual. Lowered via
    /// `lower()` and bit-exact vs the cpu_golden composition. The new
    /// token's K/V reach attention as `Op` (`Sub`) segments; the prefix
    /// cache is `Ext` (`Source`) segments — no cache round-trip.
    #[test]
    fn full_decode_layer_bit_exact() {
        let (h, hd, hq, hkv, i, l) = (16u32, 4u32, 4u32, 2u32, 32u32, 3u32);
        let (qdim, kvdim) = (hq * hd, hkv * hd); // 16, 8
        let eps = 1e-5f32;
        let scale = 1.0 / (hd as f32).sqrt();

        // Sources 0..=13.
        let res_in = rng_fill(h as usize, 101);
        let in_ln = rng_fill(h as usize, 102);
        let wq = rng_fill((qdim * h) as usize, 103);
        let wk = rng_fill((kvdim * h) as usize, 104);
        let wv = rng_fill((kvdim * h) as usize, 105);
        let cos = rng_fill(hd as usize, 106);
        let sin = rng_fill(hd as usize, 107);
        let prefix_k = rng_fill((l * kvdim) as usize, 108);
        let prefix_v = rng_fill((l * kvdim) as usize, 109);
        let wo = rng_fill((h * qdim) as usize, 110);
        let post_ln = rng_fill(h as usize, 111);
        let wgate = rng_fill((i * h) as usize, 112);
        let wup = rng_fill((i * h) as usize, 113);
        let wdown = rng_fill((h * i) as usize, 114);

        // ── Reference forward via cpu_golden ──
        let (hs, hds, hqs, hkvs, is, qd, kvd) = (
            h as usize,
            hd as usize,
            hq as usize,
            hkv as usize,
            i as usize,
            qdim as usize,
            kvdim as usize,
        );
        let mut xn = vec![0f32; hs];
        cpu_golden::rmsnorm(&res_in, &in_ln, &mut xn, eps);
        let mut q = vec![0f32; qd];
        cpu_golden::gemm(&xn, &wq, &mut q, 1, hs, qd);
        let mut k = vec![0f32; kvd];
        cpu_golden::gemm(&xn, &wk, &mut k, 1, hs, kvd);
        let mut v = vec![0f32; kvd];
        cpu_golden::gemm(&xn, &wv, &mut v, 1, hs, kvd);
        let mut q_rot = vec![0f32; qd];
        cpu_golden::rope(&q, &cos, &sin, &[0i32], 1, hqs, hds, &mut q_rot);
        let mut k_rot = vec![0f32; kvd];
        cpu_golden::rope(&k, &cos, &sin, &[0i32], 1, hkvs, hds, &mut k_rot);
        let mut k_all = prefix_k.clone();
        k_all.extend_from_slice(&k_rot);
        let mut v_all = prefix_v.clone();
        v_all.extend_from_slice(&v);
        let mut attn = vec![0f32; qd];
        cpu_golden::attention_decode(
            &q_rot,
            &k_all,
            &v_all,
            &mut attn,
            (l + 1) as usize,
            hqs,
            hkvs,
            hds,
            scale,
        );
        let mut o = vec![0f32; hs];
        cpu_golden::gemm(&attn, &wo, &mut o, 1, qd, hs);
        let mut res_mid = vec![0f32; hs];
        cpu_golden::add(&o, &res_in, &mut res_mid);
        let mut xn2 = vec![0f32; hs];
        cpu_golden::rmsnorm(&res_mid, &post_ln, &mut xn2, eps);
        let mut gate = vec![0f32; is];
        cpu_golden::gemm(&xn2, &wgate, &mut gate, 1, hs, is);
        let mut up = vec![0f32; is];
        cpu_golden::gemm(&xn2, &wup, &mut up, 1, hs, is);
        let mut act = vec![0f32; is];
        cpu_golden::fused_gate_up_silu_mul(&gate, &up, &mut act);
        let mut down = vec![0f32; hs];
        cpu_golden::gemm(&act, &wdown, &mut down, 1, is, hs);
        let mut want = vec![0f32; hs];
        cpu_golden::add(&down, &res_mid, &mut want);

        // ── The same layer as a LoweringInput ──
        let ss = |rows: u32, cols: u32| SourceShape { rows, cols };
        let input = LoweringInput {
            sources: vec![
                ss(1, h),     // 0 res_in
                ss(1, h),     // 1 input_ln
                ss(qdim, h),  // 2 Wq
                ss(kvdim, h), // 3 Wk
                ss(kvdim, h), // 4 Wv
                ss(1, hd),    // 5 cos
                ss(1, hd),    // 6 sin
                ss(l, kvdim), // 7 prefix_k
                ss(l, kvdim), // 8 prefix_v
                ss(h, qdim),  // 9 Wo
                ss(1, h),     // 10 post_ln
                ss(i, h),     // 11 Wgate
                ss(i, h),     // 12 Wup
                ss(h, i),     // 13 Wdown
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::RmsNorm { eps },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: qdim, k: h },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim, k: h },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(3)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim, k: h },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(4)],
                },
                OpDesc {
                    op: LoweredOp::RopeRotate { head_dim: hd },
                    m: 1,
                    inputs: vec![InputRef::Op(1), InputRef::Ext(5), InputRef::Ext(6)],
                },
                OpDesc {
                    op: LoweredOp::RopeRotate { head_dim: hd },
                    m: 1,
                    inputs: vec![InputRef::Op(2), InputRef::Ext(5), InputRef::Ext(6)],
                },
                OpDesc {
                    op: LoweredOp::AttnDecode {
                        num_q_heads: hq,
                        num_kv_heads: hkv,
                        head_dim: hd,
                        scale,
                    },
                    m: 1,
                    // Q_rot, (prefix_k, prefix_v), (k_rot, v_new)
                    inputs: vec![
                        InputRef::Op(4),
                        InputRef::Ext(7),
                        InputRef::Ext(8),
                        InputRef::Op(5),
                        InputRef::Op(3),
                    ],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: h, k: qdim },
                    m: 1,
                    inputs: vec![InputRef::Op(6), InputRef::Ext(9)],
                },
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(7), InputRef::Ext(0)],
                },
                OpDesc {
                    op: LoweredOp::RmsNorm { eps },
                    m: 1,
                    inputs: vec![InputRef::Op(8), InputRef::Ext(10)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: i, k: h },
                    m: 1,
                    inputs: vec![InputRef::Op(9), InputRef::Ext(11)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: i, k: h },
                    m: 1,
                    inputs: vec![InputRef::Op(9), InputRef::Ext(12)],
                },
                OpDesc {
                    op: LoweredOp::Silu,
                    m: 1,
                    inputs: vec![InputRef::Op(10)],
                },
                OpDesc {
                    op: LoweredOp::Mul,
                    m: 1,
                    inputs: vec![InputRef::Op(12), InputRef::Op(11)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: h, k: i },
                    m: 1,
                    inputs: vec![InputRef::Op(13), InputRef::Ext(13)],
                },
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(14), InputRef::Op(8)],
                },
            ],
            result: 15,
        };

        let g = lower(&input);
        let srcs: Vec<&[f32]> = vec![
            &res_in, &in_ln, &wq, &wk, &wv, &cos, &sin, &prefix_k, &prefix_v, &wo, &post_ln,
            &wgate, &wup, &wdown,
        ];
        let got = assemble_result(&g, &eval_dag(&g, &srcs));
        assert_eq!(got, want, "full decode layer must be bit-exact");

        // Tier A: the whole layer also replays bit-exact through the tape
        // player — partitioned across workers, with the rope→attn, gemm,
        // and both residual edges crossing workers as p2p Wait/Signal.
        for &p in &[1u32, 3, 8] {
            let sched = crate::tape::partition_roundrobin(&g, p);
            let played = assemble_result(&g, &crate::tape::play(&g, &sched, &srcs));
            assert_eq!(played, want, "full decode layer tape replay p={p}");
        }

        // And via the real cost-aware wavefront scheduler (cost = output
        // elements as a stand-in for the injected µs cost): the whole
        // layer, partitioned and scheduled, still replays bit-exact.
        let cost = |node: &crate::subtile::SubtileNode| (node.out_rows * node.out_cols) as f64;
        for &p in &[2u32, 4, 8] {
            let s = crate::scheduler::schedule_wavefront(
                &g,
                cost,
                crate::scheduler::ScheduleParams {
                    num_workers: p,
                    wait_cost_us: 0.18,
                },
            );
            let played = assemble_result(&g, &crate::tape::play(&g, &s, &srcs));
            assert_eq!(
                played, want,
                "full decode layer via wavefront scheduler p={p}"
            );
        }
    }

    /// A WHOLE Llama-3.2-style decode forward (M=1): `embed` as a
    /// read-only `Source` (the runtime gathers the token row before the
    /// kernel — embed is a cheap host-side lookup, not a megakernel op),
    /// then `NL` stacked decode layers (each input-norm → q/k/v → rope →
    /// GQA attn → o-proj → residual → post-norm → SwiGLU → residual), a
    /// final RMS-norm, and the `lm_head` GEMM to logits (argmax lives
    /// outside the graph). Bit-exact vs the cpu_golden composition.
    ///
    /// This is exactly the structure the macro→wavefront bridge
    /// (`ferrite-forward-macro::to_wavefront`) targets when it walks a
    /// solved Llama FUF: the embed tile becomes `sources[0]`, every
    /// `rope_append` splits into two `RopeRotate`s with the un-roped V
    /// aliasing the V-proj output, `attention` reads the prefix cache as
    /// `Source` segments + the new token as `Sub` edges, the cross-layer
    /// residual stream chains via `Op` refs, and `lm_head` is the result.
    /// Locking it here (Mac-testable) de-risks that bridge, whose own
    /// crate can't `cargo test` on this host.
    #[test]
    fn full_forward_bit_exact() {
        let (h, hd, hq, hkv, i, l, nl, vocab) =
            (16u32, 4u32, 4u32, 2u32, 32u32, 3u32, 2usize, 12u32);
        let (qdim, kvdim) = (hq * hd, hkv * hd);
        let eps = 1e-5f32;
        let scale = 1.0 / (hd as f32).sqrt();
        let (hs, hds, hqs, hkvs, is, qd, kvd, ls) = (
            h as usize,
            hd as usize,
            hq as usize,
            hkv as usize,
            i as usize,
            qdim as usize,
            kvdim as usize,
            l as usize,
        );

        // ── Source layout: 0 = embedded hidden, then 13 per layer,
        //    then final norm + lm_head. ──
        let per_layer = 13usize;
        let embed = rng_fill(hs, 7); // the gathered token row
        // Per-layer source buffers, distinct seeds per (layer, slot).
        let mk =
            |layer: usize, slot: usize, n: usize| rng_fill(n, (1000 + layer * 100 + slot) as u64);
        struct LayerW {
            in_ln: Vec<f32>,
            wq: Vec<f32>,
            wk: Vec<f32>,
            wv: Vec<f32>,
            cos: Vec<f32>,
            sin: Vec<f32>,
            pk: Vec<f32>,
            pv: Vec<f32>,
            wo: Vec<f32>,
            post_ln: Vec<f32>,
            wgate: Vec<f32>,
            wup: Vec<f32>,
            wdown: Vec<f32>,
        }
        let layers: Vec<LayerW> = (0..nl)
            .map(|ly| LayerW {
                in_ln: mk(ly, 0, hs),
                wq: mk(ly, 1, qd * hs),
                wk: mk(ly, 2, kvd * hs),
                wv: mk(ly, 3, kvd * hs),
                cos: mk(ly, 4, hds),
                sin: mk(ly, 5, hds),
                pk: mk(ly, 6, ls * kvd),
                pv: mk(ly, 7, ls * kvd),
                wo: mk(ly, 8, hs * qd),
                post_ln: mk(ly, 9, hs),
                wgate: mk(ly, 10, is * hs),
                wup: mk(ly, 11, is * hs),
                wdown: mk(ly, 12, hs * is),
            })
            .collect();
        let final_norm = rng_fill(hs, 9001);
        let lm_head = rng_fill((vocab * h) as usize, 9002);

        // ── Reference forward via cpu_golden ──
        let mut hidden = embed.clone();
        for ly in &layers {
            let mut xn = vec![0f32; hs];
            cpu_golden::rmsnorm(&hidden, &ly.in_ln, &mut xn, eps);
            let mut q = vec![0f32; qd];
            cpu_golden::gemm(&xn, &ly.wq, &mut q, 1, hs, qd);
            let mut k = vec![0f32; kvd];
            cpu_golden::gemm(&xn, &ly.wk, &mut k, 1, hs, kvd);
            let mut v = vec![0f32; kvd];
            cpu_golden::gemm(&xn, &ly.wv, &mut v, 1, hs, kvd);
            let mut q_rot = vec![0f32; qd];
            cpu_golden::rope(&q, &ly.cos, &ly.sin, &[0i32], 1, hqs, hds, &mut q_rot);
            let mut k_rot = vec![0f32; kvd];
            cpu_golden::rope(&k, &ly.cos, &ly.sin, &[0i32], 1, hkvs, hds, &mut k_rot);
            let mut k_all = ly.pk.clone();
            k_all.extend_from_slice(&k_rot);
            let mut v_all = ly.pv.clone();
            v_all.extend_from_slice(&v);
            let mut attn = vec![0f32; qd];
            cpu_golden::attention_decode(
                &q_rot,
                &k_all,
                &v_all,
                &mut attn,
                (l + 1) as usize,
                hqs,
                hkvs,
                hds,
                scale,
            );
            let mut o = vec![0f32; hs];
            cpu_golden::gemm(&attn, &ly.wo, &mut o, 1, qd, hs);
            let mut res_mid = vec![0f32; hs];
            cpu_golden::add(&o, &hidden, &mut res_mid);
            let mut xn2 = vec![0f32; hs];
            cpu_golden::rmsnorm(&res_mid, &ly.post_ln, &mut xn2, eps);
            let mut gate = vec![0f32; is];
            cpu_golden::gemm(&xn2, &ly.wgate, &mut gate, 1, hs, is);
            let mut up = vec![0f32; is];
            cpu_golden::gemm(&xn2, &ly.wup, &mut up, 1, hs, is);
            let mut act = vec![0f32; is];
            cpu_golden::fused_gate_up_silu_mul(&gate, &up, &mut act);
            let mut down = vec![0f32; hs];
            cpu_golden::gemm(&act, &ly.wdown, &mut down, 1, is, hs);
            let mut new_hidden = vec![0f32; hs];
            cpu_golden::add(&down, &res_mid, &mut new_hidden);
            hidden = new_hidden;
        }
        let mut normed = vec![0f32; hs];
        cpu_golden::rmsnorm(&hidden, &final_norm, &mut normed, eps);
        let mut want = vec![0f32; vocab as usize];
        cpu_golden::gemm(&normed, &lm_head, &mut want, 1, hs, vocab as usize);

        // ── The same forward as a LoweringInput ──
        // sources[0] = embed; then per-layer blocks; then norm, lm_head.
        let ss = |rows: u32, cols: u32| SourceShape { rows, cols };
        let mut sources = vec![ss(1, h)]; // 0: embedded hidden
        let mut flat: Vec<&[f32]> = vec![&embed];
        for ly in &layers {
            sources.extend_from_slice(&[
                ss(1, h),
                ss(qdim, h),
                ss(kvdim, h),
                ss(kvdim, h),
                ss(1, hd),
                ss(1, hd),
                ss(l, kvdim),
                ss(l, kvdim),
                ss(h, qdim),
                ss(1, h),
                ss(i, h),
                ss(i, h),
                ss(h, i),
            ]);
            flat.extend_from_slice(&[
                &ly.in_ln,
                &ly.wq,
                &ly.wk,
                &ly.wv,
                &ly.cos,
                &ly.sin,
                &ly.pk,
                &ly.pv,
                &ly.wo,
                &ly.post_ln,
                &ly.wgate,
                &ly.wup,
                &ly.wdown,
            ]);
        }
        let norm_src = sources.len();
        sources.push(ss(1, h));
        flat.push(&final_norm);
        let lm_head_src = sources.len();
        sources.push(ss(vocab, h));
        flat.push(&lm_head);

        // Emit ops layer by layer; `res_in` chains the residual stream.
        let mut ops: Vec<OpDesc> = Vec::new();
        let mut res_in = InputRef::Ext(0); // embed feeds layer 0's residual
        for ly_idx in 0..nl {
            let s0 = 1 + ly_idx * per_layer; // first source of this layer
            let e = |off: usize| InputRef::Ext(s0 + off);
            let base = ops.len();
            let op = |i: usize| InputRef::Op(base + i);
            ops.push(OpDesc {
                op: LoweredOp::RmsNorm { eps },
                m: 1,
                inputs: vec![res_in, e(0)],
            });
            ops.push(OpDesc {
                op: LoweredOp::Gemm { n: qdim, k: h },
                m: 1,
                inputs: vec![op(0), e(1)],
            });
            ops.push(OpDesc {
                op: LoweredOp::Gemm { n: kvdim, k: h },
                m: 1,
                inputs: vec![op(0), e(2)],
            });
            ops.push(OpDesc {
                op: LoweredOp::Gemm { n: kvdim, k: h },
                m: 1,
                inputs: vec![op(0), e(3)],
            });
            ops.push(OpDesc {
                op: LoweredOp::RopeRotate { head_dim: hd },
                m: 1,
                inputs: vec![op(1), e(4), e(5)],
            });
            ops.push(OpDesc {
                op: LoweredOp::RopeRotate { head_dim: hd },
                m: 1,
                inputs: vec![op(2), e(4), e(5)],
            });
            ops.push(OpDesc {
                op: LoweredOp::AttnDecode {
                    num_q_heads: hq,
                    num_kv_heads: hkv,
                    head_dim: hd,
                    scale,
                },
                m: 1,
                // Q_rot, (prefix_k, prefix_v), (k_rot, v_new)
                inputs: vec![op(4), e(6), e(7), op(5), op(3)],
            });
            ops.push(OpDesc {
                op: LoweredOp::Gemm { n: h, k: qdim },
                m: 1,
                inputs: vec![op(6), e(8)],
            });
            ops.push(OpDesc {
                op: LoweredOp::Add,
                m: 1,
                inputs: vec![op(7), res_in],
            });
            ops.push(OpDesc {
                op: LoweredOp::RmsNorm { eps },
                m: 1,
                inputs: vec![op(8), e(9)],
            });
            ops.push(OpDesc {
                op: LoweredOp::Gemm { n: i, k: h },
                m: 1,
                inputs: vec![op(9), e(10)],
            });
            ops.push(OpDesc {
                op: LoweredOp::Silu,
                m: 1,
                inputs: vec![op(10)],
            });
            ops.push(OpDesc {
                op: LoweredOp::Gemm { n: i, k: h },
                m: 1,
                inputs: vec![op(9), e(11)],
            });
            ops.push(OpDesc {
                op: LoweredOp::Mul,
                m: 1,
                inputs: vec![op(11), op(12)],
            });
            ops.push(OpDesc {
                op: LoweredOp::Gemm { n: h, k: i },
                m: 1,
                inputs: vec![op(13), e(12)],
            });
            ops.push(OpDesc {
                op: LoweredOp::Add,
                m: 1,
                inputs: vec![op(14), op(8)],
            });
            res_in = InputRef::Op(base + 15); // this layer's residual out
        }
        // Final norm + lm_head.
        let fn_idx = ops.len();
        ops.push(OpDesc {
            op: LoweredOp::RmsNorm { eps },
            m: 1,
            inputs: vec![res_in, InputRef::Ext(norm_src)],
        });
        ops.push(OpDesc {
            op: LoweredOp::Gemm { n: vocab, k: h },
            m: 1,
            inputs: vec![InputRef::Op(fn_idx), InputRef::Ext(lm_head_src)],
        });
        let result = ops.len() - 1;

        let input = LoweringInput {
            sources,
            ops,
            result,
        };
        let g = lower(&input);
        let got = assemble_result(&g, &eval_dag(&g, &flat));
        assert_eq!(
            got, want,
            "full forward (embed→{nl} layers→norm→lm_head) must be bit-exact"
        );

        // Tier A: the whole forward also replays bit-exact through the
        // tape player and the wavefront scheduler across worker counts.
        for &p in &[1u32, 4, 8] {
            let sched = crate::tape::partition_roundrobin(&g, p);
            let played = assemble_result(&g, &crate::tape::play(&g, &sched, &flat));
            assert_eq!(played, want, "full forward tape replay p={p}");
        }
        let cost = |node: &crate::subtile::SubtileNode| (node.out_rows * node.out_cols) as f64;
        for &p in &[2u32, 4, 8] {
            let s = crate::scheduler::schedule_wavefront(
                &g,
                cost,
                crate::scheduler::ScheduleParams {
                    num_workers: p,
                    wait_cost_us: 0.18,
                },
            );
            let played = assemble_result(&g, &crate::tape::play(&g, &s, &flat));
            assert_eq!(played, want, "full forward via wavefront scheduler p={p}");
        }
    }
}
