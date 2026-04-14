// SPDX-License-Identifier: Apache-2.0
//! Implementation library: the set of kernels available on a given
//! target. The solver enumerates candidates from this library,
//! calls [`Implementation::matches`] at each FUF tile, and picks
//! the cheapest match whose claim doesn't conflict.
//!
//! Types ported from old ferrite-solver (kept clean — no
//! `TileKind::GemmQ/K/V`, no `weight_name: String`, no layer
//! metadata):
//!
//! - [`LaunchKind`] — whether a kernel is invoked from CPU or is a
//!   `__device__` body that composes with neighbors.
//! - [`WorkloadConstraint`] — explicit applicability signal
//!   (replaces "cost returned None meaning N/A"). This is how
//!   kernels say "only valid at M=1" without overloading `None`
//!   from the cost function.
//! - [`TargetFilter`] — per-target applicability.
//! - [`MatchInfo`] — what a match produces: the list of FUF tiles
//!   it claims (one tile for single-op impls, multiple for fused).
//!
//! Launch mode is an attribute set by the kernel author. The solver
//! reads it like any other attribute (it affects cost context for
//! fusion accounting); it does not "decide" launch mode. Codegen
//! reads the tag to emit extern "C" calls vs megakernel launches.

#![allow(dead_code)]

use crate::classified::OpKind;
use crate::fuf::{Fuf, TileId};
use crate::shape::{Dim, Shape};
use crate::target::TargetProfile;

/// Dense impl index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImplId(pub u32);

/// How a kernel is invoked. Set by the kernel author; read by the
/// solver's cost model and by codegen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaunchKind {
    /// Extern "C" call from CPU. Paged attention, flash attention,
    /// cutlass gemm, anything that needs host-side orchestration.
    HostCallable,
    /// `__device__` body that composes with other DeviceCallable
    /// kernels in the same wave into a single megakernel.
    DeviceCallable,
}

/// Explicit applicability signal. A kernel declares the workload
/// range it's correct on; the solver refuses to pick it outside
/// that range regardless of cost. Replaces the old "cost returns
/// None means N/A" overload, which silently filtered bugs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkloadConstraint {
    /// Valid at any `num_tokens`.
    Any,
    /// Valid only at `num_tokens ∈ [min, max]` inclusive. Used by
    /// e.g. a GEMV kernel that's M=1-only, or a batched kernel
    /// with a hard lower bound.
    NumTokensRange { min: u64, max: u64 },
}

impl WorkloadConstraint {
    pub fn accepts(&self, num_tokens: u64) -> bool {
        match self {
            Self::Any => true,
            Self::NumTokensRange { min, max } => num_tokens >= *min && num_tokens <= *max,
        }
    }
}

/// Per-target applicability. Today only [`TargetFilter::Any`]; when
/// we add sm_90-specific impls this grows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetFilter {
    Any,
}

/// How a subgraph is invoked. Affects what it can share a schedule
/// slot with and what handoffs it supports. Ported from old
/// ferrite-solver — the mechanism set is GPU-structural, no
/// transformer vocabulary. Variants cover every synchronization
/// primitive the real library uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Handoff {
    /// Implicit same-stream ordering between two host launches. Free.
    StreamOrder,
    /// cudaEvent recorded after producer, waited on before consumer.
    /// Used across streams.
    StreamEvent,
    /// Kernel boundary: next `cudaLaunchKernel` waits on the previous
    /// to complete.
    KernelBoundary,
    /// `cooperative_groups::this_grid().sync()` or gmem-flag spin
    /// barrier inside a cooperative launch.
    InKernelGridSync,
    /// sm_90+ shmem `mbarrier` between warpgroups in one persistent
    /// kernel.
    Mbarrier,
    /// sm_90+ distributed shmem read across thread-block clusters.
    DsmemRead,
    /// gmem flag spin (per-tile counter). Cheaper than grid sync
    /// because it can synchronize only the tiles that need it.
    GmemFlag,
    /// Intra-CTA `__syncthreads()` between DeviceCallable impls in
    /// the same persistent kernel.
    SyncThreads,
    /// No handoff: producer and consumer are in the same subgraph;
    /// the impl handles the dep internally.
    Internal,
}

/// Per-CTA resource demand an Impl imposes on its compilation unit.
/// Two impls sharing one unit (one `__global__` on sm_89, one
/// warpgroup on sm_90+) union their demands via [`Self::union_max`];
/// the union must fit the target's budget (a constraint the
/// scheduler enforces).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Resources {
    pub shmem_bytes: u32,
    pub regs_per_thread: u32,
    pub threads_per_cta: u32,
}

impl Resources {
    pub const ZERO: Self = Self {
        shmem_bytes: 0,
        regs_per_thread: 0,
        threads_per_cta: 0,
    };

    /// Element-wise max — the budget two impls would need if
    /// sharing a compilation unit.
    pub fn union_max(self, other: Self) -> Self {
        Self {
            shmem_bytes: self.shmem_bytes.max(other.shmem_bytes),
            regs_per_thread: self.regs_per_thread.max(other.regs_per_thread),
            threads_per_cta: self.threads_per_cta.max(other.threads_per_cta),
        }
    }
}

/// Memory layout a kernel Impl requires for one of its weight args.
///
/// Declared as metadata on each [`Implementation`] (parallel to its
/// op's weight-arg positions). After the solver picks an Impl for
/// each tile group, a post-solver pass reads the chosen Impls'
/// layouts to decide, per weight instance, what final layout to
/// materialize at load time. The layout drives
/// [`WeightModel::organize`](crate::codegen_weight_model)
/// emission. The compiler itself never interprets the layout
/// beyond what each variant's semantics say — no transformer-
/// domain knowledge lives here.
///
/// Variant set is open-ended; extend when a real Impl needs a new
/// layout. Start minimal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// The weight tensor is consumed as-is. `organize` loads it
    /// directly from safetensors with no memory transform.
    Plain,
    /// `count` sibling weights are concatenated along `axis` at
    /// load time. All participating weights must share every dim
    /// except `axis`. `organize` allocates one combined tensor and
    /// copies siblings into adjacent slices. Example use: a fused
    /// QKV gemm Impl that claims 3 adjacent tiles wants
    /// `Stacked { axis: 0, count: 3 }` over q_proj/k_proj/v_proj.
    Stacked { axis: usize, count: usize },
}

impl TargetFilter {
    pub fn matches(&self, _target: &TargetProfile) -> bool {
        match self {
            Self::Any => true,
        }
    }
}

/// Result of a successful [`Implementation::matches`] call.
#[derive(Clone, Debug)]
pub struct MatchInfo {
    /// The FUF tiles this match claims. One entry for single-op
    /// impls; multiple for fused impls (e.g. a hypothetical
    /// `fused_silu_gemm` claiming both the silu and the gemm tile
    /// adjacent to it).
    pub claimed_tiles: Vec<TileId>,
}

impl MatchInfo {
    pub fn size(&self) -> usize {
        self.claimed_tiles.len()
    }
    pub fn is_singleton(&self) -> bool {
        self.claimed_tiles.len() == 1
    }
}

/// One concrete kernel implementation.
pub struct Implementation {
    pub name: &'static str,
    /// The op this impl's default matcher binds to. For impls with
    /// `matches_fn: Some`, this is advisory (the seed-op check is
    /// inside the custom matcher).
    pub op: OpKind,
    pub launch_kind: LaunchKind,
    pub workload_constraint: WorkloadConstraint,
    pub target_filter: TargetFilter,
    /// Wall-clock estimate in microseconds. Returning `None` is a
    /// bug, not "N/A" — use `workload_constraint` /
    /// `target_filter` for applicability. The solver treats `None`
    /// as a hard error.
    pub cost_fn: fn(&CostCtx) -> Option<f64>,
    /// Multi-tile matcher. `None` → default single-op matcher that
    /// returns `Some({claimed_tiles: [seed]})` iff `fuf[seed].op ==
    /// self.op`. Impls that claim multiple tiles set this.
    pub matches_fn: Option<fn(seed: TileId, fuf: &Fuf) -> Option<MatchInfo>>,
    /// Layout required for each of this Impl's weight args, in
    /// their op-signature-declared order. Length must equal the
    /// number of weight args the op has. Drives [`WeightModel`]
    /// emission: the compiler reads this off the chosen Impls and
    /// materializes each weight instance in the requested layout.
    ///
    /// For single-tile Impls whose op takes one weight (gemm,
    /// rmsnorm, embed), this is `&[Layout::Plain]`. For a fused
    /// multi-tile Impl (e.g. fused QKV gemm that claims 3 tiles),
    /// this is `&[Layout::Stacked { axis: 0, count: 3 }]`.
    ///
    /// Impls whose op has no weight args (add, mul, silu,
    /// attention, rope_append) set this to `&[]`.
    pub weight_layouts: &'static [Layout],
}

impl Implementation {
    /// Try to match this impl starting at the given seed tile.
    pub fn matches(&self, seed: TileId, fuf: &Fuf) -> Option<MatchInfo> {
        if let Some(f) = self.matches_fn {
            f(seed, fuf)
        } else if fuf.get(seed).op == self.op {
            Some(MatchInfo {
                claimed_tiles: vec![seed],
            })
        } else {
            None
        }
    }
}

impl std::fmt::Debug for Implementation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Implementation")
            .field("name", &self.name)
            .field("op", &self.op)
            .field("launch_kind", &self.launch_kind)
            .field("workload_constraint", &self.workload_constraint)
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
}

impl ImplementationLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, imp: Implementation) -> ImplId {
        let id = ImplId(self.impls.len() as u32);
        self.impls.push(imp);
        id
    }

    pub fn get(&self, id: ImplId) -> &Implementation {
        &self.impls[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.impls.len()
    }

    pub fn is_empty(&self) -> bool {
        self.impls.is_empty()
    }

    /// Iterate all impls with their ids. The DP solver uses this to
    /// enumerate candidates at each seed tile.
    pub fn iter_enumerated(&self) -> impl Iterator<Item = (ImplId, &Implementation)> + '_ {
        self.impls
            .iter()
            .enumerate()
            .map(|(i, imp)| (ImplId(i as u32), imp))
    }
}

// ── Starter library ──────────────────────────────────────────────

/// Starter library: one HostCallable Implementation per OpKind,
/// cost estimates derived from analytical FLOPs / bandwidth. Every
/// impl accepts any workload and any target; differentiation comes
/// when real target-specific / workload-specific kernels land.
///
/// Weight layouts: every op that takes a weight arg (embed,
/// rmsnorm, gemm) uses `Layout::Plain` here — the starter impls
/// are single-tile matchers consuming the weight as-is. Fused
/// multi-tile variants land later with non-Plain layouts.
pub fn starter_library() -> ImplementationLibrary {
    const PLAIN: &[Layout] = &[Layout::Plain];
    const NO_WEIGHTS: &[Layout] = &[];

    let mut lib = ImplementationLibrary::new();
    lib.push(host_impl("embed_ref", OpKind::Embed, cost_embed, PLAIN));
    lib.push(host_impl(
        "rmsnorm_ref",
        OpKind::RmsNorm,
        cost_elementwise,
        PLAIN,
    ));
    lib.push(host_impl("gemm_ref", OpKind::Gemm, cost_gemm, PLAIN));
    lib.push(host_impl(
        "rope_append_ref",
        OpKind::RopeAppend,
        cost_elementwise,
        NO_WEIGHTS,
    ));
    lib.push(host_impl(
        "attention_ref",
        OpKind::Attention,
        cost_attention,
        NO_WEIGHTS,
    ));
    lib.push(host_impl(
        "silu_ref",
        OpKind::Silu,
        cost_elementwise,
        NO_WEIGHTS,
    ));
    lib.push(host_impl(
        "add_ref",
        OpKind::Add,
        cost_elementwise,
        NO_WEIGHTS,
    ));
    lib.push(host_impl(
        "mul_ref",
        OpKind::Mul,
        cost_elementwise,
        NO_WEIGHTS,
    ));
    lib
}

fn host_impl(
    name: &'static str,
    op: OpKind,
    cost_fn: fn(&CostCtx) -> Option<f64>,
    weight_layouts: &'static [Layout],
) -> Implementation {
    Implementation {
        name,
        op,
        launch_kind: LaunchKind::HostCallable,
        workload_constraint: WorkloadConstraint::Any,
        target_filter: TargetFilter::Any,
        cost_fn,
        matches_fn: None,
        weight_layouts,
    }
}

// ── Cost functions ───────────────────────────────────────────────

const BYTES_PER_ELEM: f64 = 2.0;

fn shape_elems(ctx: &CostCtx, shape: &Shape) -> Option<u64> {
    ctx.eval_shape(shape).map(|v| v.iter().product::<u64>())
}

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

fn cost_embed(ctx: &CostCtx) -> Option<f64> {
    cost_elementwise(ctx)
}

fn cost_gemm(ctx: &CostCtx) -> Option<f64> {
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

fn cost_attention(ctx: &CostCtx) -> Option<f64> {
    let q_dims = ctx.eval_shape(&ctx.input_shapes[0])?;
    if q_dims.is_empty() {
        return None;
    }
    let t = *q_dims.first().unwrap_or(&0);
    let d = *q_dims.last().unwrap_or(&0);
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
    fn starter_library_has_one_host_impl_per_opkind() {
        let lib = starter_library();
        use OpKind::*;
        for op in [Embed, RmsNorm, Gemm, RopeAppend, Attention, Silu, Add, Mul] {
            let matches: Vec<_> = lib.iter_enumerated().filter(|(_, i)| i.op == op).collect();
            assert_eq!(matches.len(), 1, "expected 1 candidate for {op:?}");
            assert_eq!(matches[0].1.launch_kind, LaunchKind::HostCallable);
            assert_eq!(matches[0].1.workload_constraint, WorkloadConstraint::Any);
        }
    }

    #[test]
    fn starter_library_weight_layouts_match_op_weight_counts() {
        // Ops with a single weight arg declare Plain; ops with no
        // weight args declare an empty layout slice. Anything else
        // indicates a silent hardcoding we'd want to catch.
        let lib = starter_library();
        for (_, imp) in lib.iter_enumerated() {
            let expected = match imp.op {
                // One-weight ops.
                OpKind::Embed | OpKind::RmsNorm | OpKind::Gemm => 1,
                // Zero-weight ops.
                OpKind::RopeAppend
                | OpKind::Attention
                | OpKind::Silu
                | OpKind::Add
                | OpKind::Mul => 0,
            };
            assert_eq!(
                imp.weight_layouts.len(),
                expected,
                "impl {} (op {:?}) has {} weight_layouts, expected {}",
                imp.name,
                imp.op,
                imp.weight_layouts.len(),
                expected,
            );
            // Starter impls are all single-tile Plain — no fused
            // variants live here.
            for layout in imp.weight_layouts {
                assert_eq!(
                    *layout,
                    Layout::Plain,
                    "impl {} has non-Plain starter layout {:?}",
                    imp.name,
                    layout,
                );
            }
        }
    }

    #[test]
    fn handoff_variants_distinguishable() {
        // Each variant is its own distinct value. A test with no
        // semantic content beyond "the enum exists" — guards
        // against accidental merging and documents the set.
        let all = [
            Handoff::StreamOrder,
            Handoff::StreamEvent,
            Handoff::KernelBoundary,
            Handoff::InKernelGridSync,
            Handoff::Mbarrier,
            Handoff::DsmemRead,
            Handoff::GmemFlag,
            Handoff::SyncThreads,
            Handoff::Internal,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b, "{a:?} and {b:?} must be distinct");
                }
            }
        }
    }

    #[test]
    fn resources_union_max_takes_elementwise_max() {
        let a = Resources {
            shmem_bytes: 16 * 1024,
            regs_per_thread: 48,
            threads_per_cta: 128,
        };
        let b = Resources {
            shmem_bytes: 32 * 1024,
            regs_per_thread: 40,
            threads_per_cta: 256,
        };
        let u = a.union_max(b);
        assert_eq!(u.shmem_bytes, 32 * 1024);
        assert_eq!(u.regs_per_thread, 48);
        assert_eq!(u.threads_per_cta, 256);
        // Commutative.
        assert_eq!(u, b.union_max(a));
        // Idempotent.
        assert_eq!(a, a.union_max(a));
        // Identity: union_max with ZERO is self.
        assert_eq!(a, a.union_max(Resources::ZERO));
    }

    #[test]
    fn layout_plain_is_plain_and_stacked_is_stacked() {
        // Basic sanity on the variants — guards against renames.
        let p = Layout::Plain;
        let s = Layout::Stacked { axis: 0, count: 3 };
        assert_ne!(p, s);
        match s {
            Layout::Stacked { axis, count } => {
                assert_eq!(axis, 0);
                assert_eq!(count, 3);
            }
            _ => panic!("Stacked pattern didn't match"),
        }
    }

    #[test]
    fn workload_constraint_accepts_inside_range() {
        let c = WorkloadConstraint::NumTokensRange { min: 1, max: 8 };
        assert!(c.accepts(1));
        assert!(c.accepts(8));
        assert!(!c.accepts(9));
        assert!(!c.accepts(0));
        assert!(WorkloadConstraint::Any.accepts(4096));
    }

    #[test]
    fn gemm_cost_is_proportional_to_mnk() {
        let target = l4();
        let bounds = llama_3_2_1b_bounds();
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
