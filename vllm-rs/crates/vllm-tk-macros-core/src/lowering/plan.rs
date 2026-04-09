// SPDX-License-Identifier: Apache-2.0
//! Reified `ExecutionPlan` — the output of the lowering solver.
//!
//! An `ExecutionPlan` partitions a `WaveSchedule` into kernel groups,
//! where each group is the set of waves that will be lowered into one
//! `__global__` function on the target. The plan also captures the
//! per-group resource budgets (registers, shmem) the codegen will
//! use, and the handoff mechanism between consecutive groups.
//!
//! **The plan is target-independent in shape**, but its **contents**
//! are determined by the solver's reading of `LoweringConstraints`.
//! On sm_89 (where `barrier_cost_us ≫ launch_cost_us` and there is no
//! `setmaxnreg`), the solver outputs one group per wave-of-distinct-kind
//! → many groups, each launched separately. On sm_90+ (where
//! `mbarrier_handoff_us < launch_cost_us` and `regs_dynamic_per_warpgroup
//! = true`), the solver merges everything into one group → a single
//! warp-specialized persistent megakernel.
//!
//! The codegen backends (`backend_sm89`, `backend_sm90`, `backend_sm100`)
//! are pure translators from `ExecutionPlan` to target syntax — they
//! make no grouping or budget decisions of their own.

use std::collections::BTreeSet;
use std::fmt;

/// Index of a wave in the parent `WaveSchedule::waves`.
pub type WaveId = u32;

/// One kernel group in the lowered plan. Each group becomes one
/// `__global__` function on the target. Groups are executed in
/// `ExecutionPlan::groups` order; the order is dependency-safe by
/// construction (the solver only ever groups contiguous waves from
/// the BSP schedule, so cross-wave dependencies are preserved).
#[derive(Clone, Debug)]
pub struct KernelGroup {
    /// Stable id within the parent `ExecutionPlan`. Codegen uses this
    /// as the suffix on the emitted `__global__` name.
    pub group_id: u32,
    /// Contiguous slice of `WaveSchedule::waves` indices this group
    /// handles. Always non-empty.
    pub waves: Vec<WaveId>,
    /// Distinct `BoundKernel::kind()` strings present in any wave of
    /// this group. For monomorphic per-wave groups (the sm_89 norm),
    /// this is a singleton.
    pub kinds: BTreeSet<&'static str>,
    /// `max(wave.regs_per_thread)` over the waves in this group. The
    /// per-`__global__` `__launch_bounds__` driver. NVCC compiles
    /// the group with this register cap.
    pub max_regs_per_thread: u32,
    /// `max(wave.shmem_bytes)` over the waves in this group. Drives
    /// `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` for the
    /// group's launcher and the static arena sizing in the emitted
    /// kernel.
    pub max_shmem_per_cta: u32,
    /// CTA threads — currently always 256 across the megakernel
    /// template's instantiations. Stored explicitly so future variants
    /// with different CTA shapes don't require a separate plumb.
    pub threads_per_cta: u32,
    /// Estimated resident blocks per SM under this group's resource
    /// budget, derived from `max_regs_per_thread`, `max_shmem_per_cta`,
    /// and the target's per-SM hardware limits. Used by
    /// `predicted_compute_us` and the dump for human inspection.
    pub blocks_per_sm_estimate: u32,
    /// Number of in-kernel handoffs *within* this group. Equals
    /// `waves.len() - 1` for groups with more than one wave (each
    /// adjacent pair within the group shares an in-kernel barrier or
    /// mbarrier handoff). Zero for singleton groups.
    pub in_group_handoffs: u32,
    /// How control passes from this group to the next group in the
    /// `ExecutionPlan`. The last group has `Handoff::None`.
    pub handoff_to_next: Handoff,
    /// Solver's estimated wall-clock cost (microseconds) for this
    /// group: sum of effective compute over its waves (with
    /// occupancy penalty applied for any reg-budget mismatch) plus
    /// `in_group_handoffs * in_group_handoff_cost_us`.
    pub predicted_compute_us: f64,
}

/// How control passes between two consecutive `KernelGroup`s.
///
/// On sm_89 the only realized variants are `LaunchBoundary` (between
/// groups) and `InKernelGridSync` (between waves *inside* a group).
/// The sm_90 / sm_100 variants are reserved for the future warp-
/// specialized lowerings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Handoff {
    /// `cudaLaunchCooperativeKernel` boundary on the same stream.
    /// Implicit grid barrier; the next launch can't start until the
    /// previous one's grid completes. Cost in microseconds.
    LaunchBoundary { cost_us: f32 },
    /// In-kernel `cooperative_groups::this_grid().sync()` (or the
    /// gmem-flag spin barrier — measured-equivalent on L4). Used for
    /// transitions *inside* a multi-wave group on sm_89.
    InKernelGridSync { cost_us: f32 },
    /// shmem `mbarrier` handoff between two warpgroups in a single
    /// persistent `__global__`. Cost in microseconds. sm_90+ only.
    MbarrierShmem { cost_us: f32 },
    /// Distributed-shmem cluster handoff (sm_90+). Cost in microseconds.
    /// Reserved for future fused-cluster passes.
    DsmemCluster { cost_us: f32 },
    /// No handoff — last group in the plan.
    None,
}

impl Handoff {
    /// Wall-clock cost in microseconds. `Handoff::None` is 0.
    pub fn cost_us(&self) -> f32 {
        match self {
            Handoff::LaunchBoundary { cost_us } => *cost_us,
            Handoff::InKernelGridSync { cost_us } => *cost_us,
            Handoff::MbarrierShmem { cost_us } => *cost_us,
            Handoff::DsmemCluster { cost_us } => *cost_us,
            Handoff::None => 0.0,
        }
    }
}

/// The complete lowered plan for one `WaveSchedule` on one target.
#[derive(Clone, Debug)]
pub struct ExecutionPlan {
    /// The kernel groups, in execution order. Always non-empty for a
    /// non-empty schedule.
    pub groups: Vec<KernelGroup>,
    /// Solver's predicted wall-clock total in microseconds: sum of
    /// per-group `predicted_compute_us` plus the boundary handoff
    /// costs between consecutive groups.
    pub predicted_total_us: f64,
    /// Sum of all in-kernel grid sync costs that the legacy
    /// "single mega `__global__`" lowering would have paid for the
    /// same schedule. The diff vs the actual plan's barrier cost is
    /// the headline saving from the lowering refactor.
    pub legacy_grid_sync_total_us: f64,
}

impl ExecutionPlan {
    /// Total wall-clock saving (microseconds) of this plan vs the
    /// hypothetical legacy "all waves in one `__global__` joined by
    /// `cooperative_groups::this_grid().sync()`" lowering.
    pub fn savings_vs_legacy_us(&self) -> f64 {
        self.legacy_grid_sync_total_us
            - self
                .groups
                .iter()
                .map(|g| g.handoff_to_next.cost_us() as f64)
                .sum::<f64>()
            - self
                .groups
                .iter()
                .map(|g| g.in_group_handoffs as f64 * in_group_handoff_cost_us(g.handoff_to_next))
                .sum::<f64>()
    }
}

/// Best-effort: the in-group handoff cost matches the next-group
/// handoff mechanism (mbarrier inside a warp-specialized group, grid
/// sync inside a multi-wave sm_89 group, etc.). Used only for the
/// `savings_vs_legacy_us` rollup; the solver itself uses the
/// authoritative `LoweringConstraints` value.
fn in_group_handoff_cost_us(next_handoff: Handoff) -> f64 {
    match next_handoff {
        Handoff::MbarrierShmem { cost_us } => cost_us as f64,
        Handoff::DsmemCluster { cost_us } => cost_us as f64,
        // Default: the in-group cost is a grid sync. sm_89's case.
        _ => 100.0,
    }
}

impl fmt::Display for ExecutionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ExecutionPlan")?;
        writeln!(
            f,
            "  predicted total: {:.3} ms",
            self.predicted_total_us / 1000.0
        )?;
        writeln!(
            f,
            "  legacy grid_sync total (hypothetical): {:.3} ms",
            self.legacy_grid_sync_total_us / 1000.0
        )?;
        writeln!(
            f,
            "  savings vs legacy: {:.3} ms",
            self.savings_vs_legacy_us() / 1000.0
        )?;
        writeln!(f, "  groups: {}", self.groups.len())?;
        writeln!(f)?;
        for g in &self.groups {
            writeln!(f, "{g}")?;
        }
        Ok(())
    }
}

impl fmt::Display for KernelGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kinds: Vec<&str> = self.kinds.iter().copied().collect();
        let kind_str = kinds.join(", ");
        writeln!(
            f,
            "  Group {:>3}: {} wave(s) [{}..={}], kinds={{{}}}",
            self.group_id,
            self.waves.len(),
            self.waves.first().copied().unwrap_or(0),
            self.waves.last().copied().unwrap_or(0),
            kind_str
        )?;
        writeln!(
            f,
            "             regs/thread={:>3} threads/CTA={:>3} shmem/CTA={:>5}B blocks/SM={}",
            self.max_regs_per_thread,
            self.threads_per_cta,
            self.max_shmem_per_cta,
            self.blocks_per_sm_estimate
        )?;
        writeln!(
            f,
            "             compute={:7.2}µs  in_group_handoffs={}  handoff_to_next={}",
            self.predicted_compute_us, self.in_group_handoffs, self.handoff_to_next
        )?;
        Ok(())
    }
}

impl fmt::Display for Handoff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Handoff::LaunchBoundary { cost_us } => write!(f, "LaunchBoundary({cost_us:.1}µs)"),
            Handoff::InKernelGridSync { cost_us } => write!(f, "InKernelGridSync({cost_us:.1}µs)"),
            Handoff::MbarrierShmem { cost_us } => write!(f, "MbarrierShmem({cost_us:.1}µs)"),
            Handoff::DsmemCluster { cost_us } => write!(f, "DsmemCluster({cost_us:.1}µs)"),
            Handoff::None => write!(f, "None"),
        }
    }
}
