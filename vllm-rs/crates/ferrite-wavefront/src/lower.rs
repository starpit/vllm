// SPDX-License-Identifier: Apache-2.0
//! `LoweringInput` — flat, topologically-ordered, backend-agnostic
//! description of a solved forward.
//!
//! One [`OpDesc`] per op, each naming its kind, row count, and inputs
//! (either a prior op's output or an external source — a
//! weight/activation/extern). It is deliberately a *plain data* mirror
//! of the macro crate's internal `Fuf` + solver `Assignment`: the
//! macro crate (which alone can see those types) does the trivial
//! structural translation and hands the result to
//! [`crate::subtile_ir::lower_region`], so all the decomposition logic
//! lives in this Mac-testable crate.
//!
//! [`fuse_silu_mul`] is a pre-pass that fuses adjacent `Silu` → `Mul`
//! into [`LoweredOp::SiluMul`] before lowering — the GPU has only the
//! fused arm, so fusion must happen before scheduling places the pair
//! on workers.
//!
//! Slated for deletion in plan §4 commit 7 (when `to_wavefront.rs`
//! builds [`crate::subtile_ir::SubtileIR`] directly).

use crate::subtile_ir::SourceShape;

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
    /// `k` is derived from the activation's column count at lowering
    /// time — it is not a separate field. (Carrying `k` separately
    /// would require a runtime `assert_eq!(in0_cols, k, …)` to defend
    /// against producer/consumer drift; per
    /// `feedback_compile_time_or_garbage` and §5 K5, that proof lives
    /// either on the producing op's column witness or as a structural
    /// derivation — never as a runtime assert.)
    Gemm { n: u32 },
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
    /// The K-side `rope_append` for the GPU megakernel: rotate K + write the
    /// rotated K / un-rotated V into the paged KV cache for `layer`. Inputs:
    /// `[K, cos, sin, V]`. Host eval is rotation only (the cache write is
    /// GPU-only — see [`SubOp::RopeAppend`]); shape-preserving on K. The macro
    /// emits this for `rope_append`'s K slot; the Q slot stays `RopeRotate`.
    RopeAppend { head_dim: u32, layer: u32 },
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
