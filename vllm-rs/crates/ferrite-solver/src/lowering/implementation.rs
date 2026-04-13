// SPDX-License-Identifier: Apache-2.0
//! [`Implementation`] trait + supporting types.
//!
//! An [`Implementation`] is a curated entry in the kernel library.
//! It describes one way to realize a subgraph of [`TileNode`]s as
//! actual GPU code: which subgraph patterns it can claim, what its
//! launch / handoff / resource requirements are, what its predicted
//! cost is on a given target.
//!
//! ## Status — what currently exists
//!
//! - The trait + supporting types are defined here.
//! - The `library` module curates concrete entries (see
//!   [`crate::lowering::library`]).
//! - `matches`, `cost_us`, `resources`, `target_compatible`, etc.
//!   are implemented per entry, hand-written Rust over the
//!   [`crate::lowering::TileGraph`]. There is intentionally no
//!   pattern DSL (procedural matchers are simpler to write and
//!   debug for the bounded set of entries we curate).
//!
//! ## What the solver does with this
//!
//! The solver enumerates candidate (subgraph, implementation) pairs
//! by calling `impl.matches(subgraph, profile)` over feasible
//! subgraphs of the tile graph. For each match the solver computes
//! the implementation's cost and resource demand on the target,
//! checks the constraints, and decides whether to commit. The
//! decision is part of the joint [`crate::lowering::Assignment`].

use std::fmt;

use crate::lowering::tile_graph::{TileGraph, TileId, TileKind};
use crate::target_profile::TargetProfile;

/// Stable identifier for one [`Implementation`] in the
/// [`crate::lowering::ImplementationLibrary`]. Indices are dense in
/// the library's `entries` vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImplId(pub u32);

/// Per-implementation hardware resource demand.
///
/// **Per-CTA** demands (regs, shmem) determine occupancy and the
/// hard shmem-budget constraint. Two implementations grouped into
/// the same compilation unit (the same `__global__` on sm_89, the
/// same warpgroup on sm_90+ with `setmaxnreg`) **union** their
/// resource demands at the unit level — that's the constraint that
/// killed CP3's per-kind kernels when they tried to share a budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Resources {
    /// Per-CTA dynamic shmem demand, in bytes.
    pub shmem_bytes: u32,
    /// Per-thread register footprint estimate.
    pub regs_per_thread: u32,
    /// CTA size this implementation runs with.
    pub threads_per_cta: u32,
}

impl Resources {
    /// Element-wise max — the resource budget two impls would need
    /// if they shared a compilation unit. Used by the resource-union
    /// constraint.
    pub fn union_max(&self, other: &Resources) -> Resources {
        Resources {
            shmem_bytes: self.shmem_bytes.max(other.shmem_bytes),
            regs_per_thread: self.regs_per_thread.max(other.regs_per_thread),
            threads_per_cta: self.threads_per_cta.max(other.threads_per_cta),
        }
    }
}

/// How an implementation is invoked / scheduled.
///
/// This is a hard constraint on what the implementation can share a
/// schedule slot with: a `HostCallback` cannot be co-resident with
/// other host calls in the same step (the host issues calls in
/// stream order); a `CooperativeLaunch` is mutually exclusive with
/// every other CooperativeLaunch on the device; a `DeviceCallable`
/// runs inside an enclosing `__global__` and is constrained by that
/// kernel's resource budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaunchKind {
    /// Host calls a vendor library function (cuBLAS, cublasLt,
    /// FlashInfer standalone, vllm-rs's fused-op kernels) which
    /// internally launches its own grid. The host launch boundary
    /// is the synchronization point.
    HostCallback,
    /// Host launches a normal `__global__` (non-cooperative) on a
    /// stream. Multiple of these can be in flight on different
    /// streams.
    RegularLaunch,
    /// Host launches a cooperative `__global__` that grabs all SMs
    /// and excludes other cooperative grids until it finishes.
    CooperativeLaunch,
    /// A `__device__` callable invoked from inside another
    /// `__global__`'s code. Contributes to the host kernel's
    /// resource budget; cannot be launched on its own.
    DeviceCallable,
}

/// Synchronization mechanism that conveys data between two
/// implementations on a producer→consumer dependency edge.
///
/// The cost of each mechanism varies by target. The solver picks
/// the cheapest mechanism that's compatible with both the producer's
/// and the consumer's [`LaunchKind`] and supported handoff sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Handoff {
    /// Two host calls on the same stream — implicit ordering, no
    /// explicit sync. Free in cost-model terms; carries the
    /// per-launch overhead of whichever launch comes next.
    StreamOrder,
    /// `cudaEvent` recorded after the producer + waited on before
    /// the consumer. ~5 µs on L4. Used to convey deps across
    /// different streams.
    StreamEvent,
    /// Kernel boundary — the next `cudaLaunchKernel` waits for the
    /// previous to complete. Cost = `LoweringConstraints::launch_cost_us`.
    KernelBoundary,
    /// In-kernel `cooperative_groups::this_grid().sync()` or
    /// gmem-flag spin barrier. ~100 µs on L4 sm_89.
    InKernelGridSync,
    /// shmem `mbarrier` — sm_90+ only, ~ns latency. Used between
    /// warpgroups inside one persistent kernel.
    Mbarrier,
    /// Distributed-shmem cluster read — sm_90+ with thread-block
    /// clusters. Reserved for the future fused-cluster passes.
    DsmemRead,
    /// gmem flag spin (per-tile counter) — usable on any target,
    /// ~hundreds of ns to a few µs depending on contention.
    /// Cheaper than InKernelGridSync because it can synchronize
    /// only the tiles that need it.
    GmemFlag,
    /// `__syncthreads()` — intra-CTA barrier. Available on all
    /// architectures (~0.3-0.5 µs). Used between DeviceCallable ops
    /// within the same persistent kernel, especially on sm89 where
    /// mbarrier isn't the primary intra-kernel mechanism. Data goes
    /// through gmem (write → barrier → read), so it truncates.
    SyncThreads,
    /// No handoff — both impls are claimed by the same subgraph
    /// (the implementation handles the dep internally).
    Internal,
}

impl Handoff {
    /// Wall-clock cost in microseconds for this handoff on the
    /// given target. Reads from [`crate::target_profile::LoweringConstraints`].
    pub fn cost_us(&self, profile: &TargetProfile) -> f64 {
        match self {
            Handoff::StreamOrder => 0.0,
            Handoff::StreamEvent => 5.0, // empirical L4 stream event cost
            Handoff::KernelBoundary => profile.lowering.launch_cost_us as f64,
            Handoff::InKernelGridSync => profile.lowering.barrier_cost_us as f64,
            Handoff::Mbarrier => profile
                .lowering
                .mbarrier_handoff_us
                .map(|c| c as f64)
                .unwrap_or(f64::INFINITY),
            Handoff::DsmemRead => profile
                .lowering
                .dsmem_cluster_handoff_us
                .map(|c| c as f64)
                .unwrap_or(f64::INFINITY),
            Handoff::GmemFlag => 0.5, // empirical sub-µs gmem-flag spin
            Handoff::SyncThreads => profile
                .lowering
                .syncthreads_handoff_us
                .map(|c| c as f64)
                .unwrap_or(f64::INFINITY),
            Handoff::Internal => 0.0,
        }
    }

    /// Whether this handoff truncates the intermediate to storage
    /// dtype. GMEM-based handoffs truncate (the value is written as
    /// bf16/fp16/fp8 then re-read). In-kernel handoffs (Internal,
    /// Mbarrier, shmem) can preserve accumulator precision (f32).
    ///
    /// This affects numerical reproducibility: a fused plan that
    /// skips a GMEM truncation point produces slightly different
    /// output than the unfused reference (more precise, but
    /// different). The solver uses this via
    /// [`Constraint::PrecisionBounded`].
    pub fn truncates_to_storage_dtype(&self) -> bool {
        match self {
            // GMEM-transiting handoffs: data is written to and read
            // from global memory in the storage dtype.
            Handoff::StreamOrder
            | Handoff::StreamEvent
            | Handoff::KernelBoundary
            | Handoff::InKernelGridSync
            | Handoff::GmemFlag
            | Handoff::SyncThreads => true,
            // In-register / shmem handoffs: data stays in the
            // accumulator's wider dtype (f32 for bf16 tensor cores).
            Handoff::Internal | Handoff::Mbarrier | Handoff::DsmemRead => false,
        }
    }
}

/// Layout of a tile in memory. Used by the layout-compatibility
/// constraint: a producer's output layout must equal the consumer's
/// input layout, OR a layout-conversion implementation must be
/// inserted between them.
///
/// Initially small. Extended as more implementations enter the
/// library — e.g. CUTLASS sm_90 wants `SwizzledTma` layouts that
/// require an explicit conversion from `RowMajor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// `[outer, inner]` row-major contiguous bf16. The default for
    /// the existing megakernel buffers.
    RowMajorBf16,
    /// `[inner, outer]` column-major bf16. cuBLAS / CUTLASS prefer
    /// this for the B operand (the weight) of a matmul.
    ColMajorBf16,
    /// Paged KV layout: `[num_pages, page_size, num_kv_heads, head_dim]`
    /// bf16. The K/V cache uses this; FlashInfer requires it.
    PagedKvBf16,
    /// "Don't care" — the producer and consumer don't constrain
    /// the layout (e.g. for residual_add the residual is just a
    /// flat buffer that can be either layout).
    Any,
}

impl Layout {
    /// Whether two layouts are interchangeable without an explicit
    /// conversion. `Any` matches anything.
    pub fn is_compatible_with(self, other: Layout) -> bool {
        match (self, other) {
            (Layout::Any, _) | (_, Layout::Any) => true,
            (a, b) => a == b,
        }
    }
}

/// What an [`Implementation::matches`] call returns when an impl
/// can claim a particular subgraph. Carries the per-claim metadata
/// the solver / cost model / backend need.
#[derive(Clone, Debug)]
pub struct MatchInfo {
    /// The tile ids this implementation would claim if chosen.
    /// Must be a connected subgraph of the [`TileGraph`].
    pub claimed_tiles: Vec<TileId>,
    /// The boundary-input tiles this claim reads from (i.e. tiles
    /// outside `claimed_tiles` whose outputs flow in). Used by the
    /// solver to wire up handoffs.
    pub boundary_inputs: Vec<TileId>,
    /// The boundary-output tiles this claim writes to (i.e. tiles
    /// inside `claimed_tiles` whose outputs are read by tiles
    /// outside the claim).
    pub boundary_outputs: Vec<TileId>,
    /// The layer this claim operates on (for per-layer impls).
    pub layer: u16,
}

/// One curated implementation in the library.
///
/// Implementations are **not** generic — each entry corresponds to
/// a specific kernel from a specific source (cuBLAS, CUTLASS sm_80
/// multistage, ThunderKittens fused gate-up, FlashInfer standalone
/// FA-2, vllm-rs `fused_add_rms_norm_inplace`, ...).
///
/// Each impl declares the subgraph patterns it can match, its
/// resource demands, its launch kind, the handoff mechanisms it
/// supports, the layouts it requires, and its target compatibility.
/// Cost is calibrated from microbench data per shape.
/// Declarative workload eligibility for an implementation.
///
/// Distinct from `target_compatible` (which is about GPU capability,
/// e.g. sm_89 vs sm_90). A `WorkloadConstraint` expresses correctness
/// requirements on the workload itself — e.g. "this GEMV kernel only
/// handles M=1" or "this batched kernel requires num_tokens ≥ 64".
///
/// Correctness, not cost: if `accepts(num_tokens)` returns false, the
/// impl must NOT be picked at that workload, regardless of its cost.
/// The solver enforces this via the `WorkloadCompatible` constraint
/// (same shape as `TargetCompatible`).
///
/// Data, not closures — so the ILP backend can linearize each variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkloadConstraint {
    /// Valid for any `num_tokens` value. Default for most impls.
    Any,
    /// Valid only when `num_tokens` falls in this inclusive range.
    /// Used by kernels with hard correctness limits — e.g. a GEMV
    /// kernel that only handles M=1, or a fused kernel designed for
    /// a specific batch block that produces garbage at other sizes.
    NumTokensRange { min: u32, max: u32 },
}

impl WorkloadConstraint {
    /// Whether this constraint admits the given `num_tokens`.
    pub fn accepts(&self, num_tokens: u32) -> bool {
        match self {
            Self::Any => true,
            Self::NumTokensRange { min, max } => num_tokens >= *min && num_tokens <= *max,
        }
    }
}

pub trait Implementation: fmt::Debug + Send + Sync {
    /// Stable name for debug / display / cost-table keys.
    fn name(&self) -> &'static str;

    /// Whether this implementation can run on the given target.
    /// E.g. CUTLASS sm_90 warp-specialized requires sm_90+ and
    /// returns false on the L4 sm_89 profile.
    fn target_compatible(&self, profile: &TargetProfile) -> bool;

    /// Workload eligibility for this implementation.
    ///
    /// Default: accepts any `num_tokens`. Override for kernels that
    /// have hard correctness requirements on the workload shape
    /// (e.g. GEMV at M=1 only, or a fused kernel that only handles
    /// specific batch sizes).
    ///
    /// This is a correctness constraint, not a cost signal. The
    /// solver refuses to pick impls whose workload constraint is
    /// violated, regardless of cost. Use `cost_us` for soft signals.
    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    /// Try to match a subgraph rooted at the given seed tile.
    /// Returns `Some(MatchInfo)` if this implementation can claim
    /// a subgraph that includes `seed`, `None` otherwise.
    ///
    /// The matcher is procedural Rust: it inspects `tile_graph`,
    /// walks neighbors of `seed`, decides whether the local
    /// structure matches the implementation's expected pattern,
    /// and returns the claim. There is intentionally no DSL.
    ///
    /// The seed-rooted form (vs free-form subgraph enumeration)
    /// keeps the solver's enumeration tractable: the solver picks
    /// a tile in topological order and queries every implementation
    /// for matches at that seed.
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        profile: &TargetProfile,
    ) -> Option<MatchInfo>;

    /// Predicted wall-clock cost in microseconds for one invocation
    /// of this implementation on the given match, on the target.
    /// Reads from a calibration table the library populates from
    /// CP4 microbench data; falls back to an analytical estimate
    /// when the shape isn't in the table.
    fn cost_us(&self, m: &MatchInfo, profile: &TargetProfile) -> f64;

    /// Per-CTA resource demand of this implementation.
    fn resources(&self, m: &MatchInfo) -> Resources;

    /// How this implementation is launched / scheduled.
    fn launch_kind(&self) -> LaunchKind;

    /// The handoff mechanisms this implementation can use to
    /// **receive** its boundary inputs from upstream.
    fn supported_input_handoffs(&self) -> &[Handoff];

    /// The handoff mechanisms this implementation can use to
    /// **convey** its boundary outputs to downstream.
    fn supported_output_handoffs(&self) -> &[Handoff];

    /// Required input layouts, parallel to `match.boundary_inputs`.
    /// Used by the layout-compatibility constraint.
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout>;

    /// Output layouts, parallel to `match.boundary_outputs`.
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout>;

    /// Whether this implementation is **classified as compute-bound**
    /// for concurrency-modeling purposes. Compute-bound impls on the
    /// same target's tensor cores cannot productively overlap with
    /// other compute-bound impls. Memory-bound impls can.
    fn is_compute_bound(&self) -> bool {
        // Default: GEMMs are compute-bound; norm/rope/etc. are not.
        // Implementations override when their character differs.
        false
    }

    /// Whether this implementation can be claimed by the same
    /// "group" / kernel as another implementation. For
    /// HostCallback the answer is always false (each host call is
    /// its own launch boundary); for DeviceCallable it depends on
    /// resource compatibility and is checked by the resource
    /// constraint, not here.
    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool {
        // Default behavior: HostCallback / RegularLaunch / Cooperative
        // never share. DeviceCallable can share if the resource
        // constraint allows. The default returns false; overrides
        // (in DeviceCallable impls) opt into sharing.
        false
    }
}

/// Helper used by the solver to seed-enumerate matches across the
/// whole library at a given tile. Returns every (impl_id, MatchInfo)
/// pair that can claim something including `seed`.
pub(crate) fn enumerate_matches_at_seed(
    library: &[Box<dyn Implementation>],
    tile_graph: &TileGraph,
    seed: TileId,
    profile: &TargetProfile,
) -> Vec<(ImplId, MatchInfo)> {
    let mut out = Vec::new();
    for (idx, imp) in library.iter().enumerate() {
        if !imp.target_compatible(profile) {
            continue;
        }
        if let Some(m) = imp.matches(tile_graph, seed, profile) {
            out.push((ImplId(idx as u32), m));
        }
    }
    out
}

/// Sanity helpers for the [`TileKind`] / [`MatchInfo`] interaction.
impl MatchInfo {
    /// Number of tiles claimed (= "fusion size" of this match).
    pub fn size(&self) -> usize {
        self.claimed_tiles.len()
    }

    /// Whether this match consists of just the single seed tile.
    pub fn is_singleton(&self) -> bool {
        self.claimed_tiles.len() == 1
    }
}

/// Convenience: filter `claimed_tiles` to those of a given kind.
/// Used by cost / layout helpers.
pub fn tiles_of_kind<'a>(
    tile_graph: &'a TileGraph,
    claimed: &'a [TileId],
    kind: TileKind,
) -> impl Iterator<Item = TileId> + 'a {
    claimed
        .iter()
        .copied()
        .filter(move |&tid| tile_graph.nodes[tid.0 as usize].kind == kind)
}
