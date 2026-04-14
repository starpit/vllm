// SPDX-License-Identifier: Apache-2.0
//! Implementation library: the set of kernels available on a given
//! target. Keyed by [`OpKind`], many impls per op allowed. Each
//! [`Implementation`] carries a cost function the solver consults.
//!
//! This is deliberately minimal. The real ferrite-kernels crate
//! holds the actual CUDA code; this module describes the
//! *metadata* the solver needs to pick between candidates. Each
//! target crate (e.g. `ferrite_kernels::l4_sm89`) will own the
//! actual Implementation instances.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::classified::OpKind;
use crate::shape::{Dim, Shape};
use crate::target::TargetProfile;

/// Dense impl index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImplId(pub u32);

/// One concrete kernel implementation for one op kind.
pub struct Implementation {
    pub name: &'static str,
    pub op: OpKind,
    /// Bitmask of cooperative-exclusive resources this impl holds
    /// for the duration of a step. Two impls with overlapping
    /// masks can't run in the same BSP step.
    pub claim_mask: u8,
    /// Estimate kernel runtime in microseconds given input shapes
    /// and this target's hardware specs, with all bound names
    /// resolved via `bounds`. Returns `None` if shapes have
    /// unresolved dim variables the impl can't estimate.
    pub cost_fn: fn(&CostCtx) -> Option<f64>,
}

impl std::fmt::Debug for Implementation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Implementation")
            .field("name", &self.name)
            .field("op", &self.op)
            .field("claim_mask", &self.claim_mask)
            .finish()
    }
}

/// Inputs to an impl's cost function.
pub struct CostCtx<'a> {
    pub input_shapes: &'a [Shape],
    pub output_shapes: &'a [Shape],
    pub target: &'a TargetProfile,
    pub bounds: &'a std::collections::BTreeMap<String, u64>,
}

impl CostCtx<'_> {
    /// Evaluate a `Dim` to a concrete integer using the current
    /// bounds. Returns `None` if the dim contains a Var.
    pub fn eval_dim(&self, dim: &Dim) -> Option<u64> {
        match dim {
            Dim::Lit(n) => Some(*n),
            Dim::Bound(name) => self.bounds.get(name).copied(),
            Dim::Mul(cs) => cs
                .iter()
                .map(|c| self.eval_dim(c))
                .try_fold(1u64, |acc, v| v.map(|x| acc.saturating_mul(x))),
            Dim::Var(_) => None,
        }
    }

    pub fn eval_shape(&self, shape: &Shape) -> Option<Vec<u64>> {
        shape.iter().map(|d| self.eval_dim(d)).collect()
    }
}

/// The library: all available implementations for some target.
#[derive(Debug, Default)]
pub struct ImplementationLibrary {
    impls: Vec<Implementation>,
    by_op: HashMap<OpKind, Vec<ImplId>>,
}

impl ImplementationLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, imp: Implementation) -> ImplId {
        let id = ImplId(self.impls.len() as u32);
        self.by_op.entry(imp.op).or_default().push(id);
        self.impls.push(imp);
        id
    }

    pub fn get(&self, id: ImplId) -> &Implementation {
        &self.impls[id.0 as usize]
    }

    pub fn candidates(&self, op: OpKind) -> &[ImplId] {
        self.by_op.get(&op).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn len(&self) -> usize {
        self.impls.len()
    }

    pub fn is_empty(&self) -> bool {
        self.impls.is_empty()
    }
}

// ── Starter library ──────────────────────────────────────────────

/// Starter library for any target: one Implementation per OpKind,
/// cost estimates derived from analytical FLOPs / bandwidth.
///
/// These are *reference* costs, not hand-tuned CUTLASS numbers.
/// Later phases can add more impls with better costs; the solver
/// will pick whichever is cheapest. For Phase 7 MVP every tile has
/// exactly one candidate, so the solver's job is trivially "assign
/// the one available impl."
pub fn starter_library() -> ImplementationLibrary {
    let mut lib = ImplementationLibrary::new();
    lib.push(Implementation {
        name: "embed_ref",
        op: OpKind::Embed,
        claim_mask: 0,
        cost_fn: cost_embed,
    });
    lib.push(Implementation {
        name: "rmsnorm_ref",
        op: OpKind::RmsNorm,
        claim_mask: 0,
        cost_fn: cost_elementwise,
    });
    lib.push(Implementation {
        name: "gemm_ref",
        op: OpKind::Gemm,
        claim_mask: 0,
        cost_fn: cost_gemm,
    });
    lib.push(Implementation {
        name: "rope_append_ref",
        op: OpKind::RopeAppend,
        claim_mask: 0,
        cost_fn: cost_elementwise,
    });
    lib.push(Implementation {
        name: "attention_ref",
        op: OpKind::Attention,
        claim_mask: 0,
        cost_fn: cost_attention,
    });
    lib.push(Implementation {
        name: "silu_ref",
        op: OpKind::Silu,
        claim_mask: 0,
        cost_fn: cost_elementwise,
    });
    lib.push(Implementation {
        name: "add_ref",
        op: OpKind::Add,
        claim_mask: 0,
        cost_fn: cost_elementwise,
    });
    lib
}

// ── Cost functions ───────────────────────────────────────────────

/// Bytes per element. Assume FP16 throughout for the cost model.
const BYTES_PER_ELEM: f64 = 2.0;

/// Total element count of a shape (product of dims).
fn shape_elems(ctx: &CostCtx, shape: &Shape) -> Option<u64> {
    ctx.eval_shape(shape).map(|v| v.iter().product::<u64>())
}

/// Elementwise ops (rmsnorm, silu, add, rope_append): memory-bound.
/// Cost = bytes_moved / peak_bandwidth.
fn cost_elementwise(ctx: &CostCtx) -> Option<f64> {
    let bytes: u64 = ctx
        .input_shapes
        .iter()
        .chain(ctx.output_shapes)
        .map(|s| shape_elems(ctx, s).unwrap_or(0))
        .sum::<u64>()
        * BYTES_PER_ELEM as u64;
    let gb_per_sec = ctx.target.memory_bandwidth_gbps;
    Some((bytes as f64 / (gb_per_sec * 1e9)) * 1e6)
}

/// Embed: one gather per output element. Treat as bandwidth-bound.
fn cost_embed(ctx: &CostCtx) -> Option<f64> {
    cost_elementwise(ctx)
}

/// Gemm: 2*M*N*K flops, compute-bound on peak FP16 tensor cores.
fn cost_gemm(ctx: &CostCtx) -> Option<f64> {
    // inputs[0]: [.., K], inputs[1]: [K, N]. M = product of input[0]
    // dims except last.
    let x_dims = ctx.eval_shape(&ctx.input_shapes[0])?;
    let w_dims = ctx.eval_shape(&ctx.input_shapes[1])?;
    if x_dims.is_empty() || w_dims.len() != 2 {
        return None;
    }
    let m: u64 = x_dims[..x_dims.len() - 1].iter().product();
    let k = *x_dims.last().unwrap();
    let n = w_dims[1];
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let peak_flops_per_sec = ctx.target.peak_tflops_fp16 * 1e12;
    Some((flops / peak_flops_per_sec) * 1e6)
}

/// Attention: rough estimate — O(num_tokens^2 * head_dim * num_heads)
/// for the score matmul + softmax + out matmul. Falls back to
/// elementwise cost if shapes are Var-contaminated.
fn cost_attention(ctx: &CostCtx) -> Option<f64> {
    // Output: [.., num_attention_heads * head_dim]. Inputs 0..2 are q/k/v.
    let q_dims = ctx.eval_shape(&ctx.input_shapes[0])?;
    if q_dims.is_empty() {
        return None;
    }
    let t = *q_dims.first().unwrap_or(&0); // num_tokens
    let d = *q_dims.last().unwrap_or(&0); // heads * head_dim
    // Two matmuls of [t, d] × [d, t] → 2 * t^2 * d, and score × v
    // of [t, t] × [t, d] → 2 * t^2 * d. Total 4 * t^2 * d.
    let flops = 4.0 * t as f64 * t as f64 * d as f64;
    let peak_flops_per_sec = ctx.target.peak_tflops_fp16 * 1e12;
    Some((flops / peak_flops_per_sec) * 1e6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::load_file as load_target;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn l4() -> TargetProfile {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("target_profiles")
            .join("l4_sm89.json");
        load_target(&path).unwrap()
    }

    fn llama_3_2_1b_bounds() -> BTreeMap<String, u64> {
        let mut m = BTreeMap::new();
        m.insert("num_hidden_layers".into(), 16);
        m.insert("hidden_size".into(), 2048);
        m.insert("intermediate_size".into(), 8192);
        m.insert("num_attention_heads".into(), 32);
        m.insert("num_key_value_heads".into(), 8);
        m.insert("head_dim".into(), 64);
        m.insert("vocab_size".into(), 128256);
        m.insert("num_tokens".into(), 1);
        m
    }

    fn bound(name: &str) -> Dim {
        Dim::Bound(name.into())
    }

    #[test]
    fn starter_library_has_one_impl_per_opkind() {
        let lib = starter_library();
        use OpKind::*;
        for op in [Embed, RmsNorm, Gemm, RopeAppend, Attention, Silu, Add] {
            assert_eq!(
                lib.candidates(op).len(),
                1,
                "expected 1 candidate for {op:?}",
            );
        }
    }

    #[test]
    fn gemm_cost_is_proportional_to_mnk() {
        let target = l4();
        let bounds = llama_3_2_1b_bounds();
        // x: [num_tokens=1, hidden_size=2048]
        // w: [hidden_size=2048, vocab_size=128256]
        // M=1, N=128256, K=2048 → 2MNK ≈ 525M FLOPs
        let inputs = vec![
            vec![bound("num_tokens"), bound("hidden_size")],
            vec![bound("hidden_size"), bound("vocab_size")],
        ];
        let outputs = vec![vec![bound("num_tokens"), bound("vocab_size")]];
        let ctx = CostCtx {
            input_shapes: &inputs,
            output_shapes: &outputs,
            target: &target,
            bounds: &bounds,
        };
        let cost_us = cost_gemm(&ctx).expect("should estimate");
        // Expected: 525e6 FLOPs / (242e12 FLOPs/sec) * 1e6 us/sec ≈ 2.17 us.
        assert!(cost_us > 0.0);
        assert!(
            cost_us < 100.0,
            "gemm estimate {cost_us} us unreasonably slow"
        );
    }

    #[test]
    fn var_in_shape_gives_none_cost() {
        let target = l4();
        let bounds = llama_3_2_1b_bounds();
        // A shape with a Var: cost must fall back to None.
        let mut solver = crate::shape::Solver::new();
        let v = solver.fresh();
        let inputs = vec![vec![bound("num_tokens"), Dim::Var(v)], vec![]];
        let outputs = vec![vec![]];
        let ctx = CostCtx {
            input_shapes: &inputs,
            output_shapes: &outputs,
            target: &target,
            bounds: &bounds,
        };
        assert!(cost_gemm(&ctx).is_none());
    }
}
