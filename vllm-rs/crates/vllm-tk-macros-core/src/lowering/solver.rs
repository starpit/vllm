// SPDX-License-Identifier: Apache-2.0
//! DP solver: turn a `WaveSchedule` into an `ExecutionPlan` by
//! partitioning the linear wave sequence into kernel groups.
//!
//! ## The optimization
//!
//! Each wave in the schedule has compute cost (in mma units → µs via
//! the target's GPU clock) and resource demand (`regs_per_thread`,
//! `shmem_bytes`) read from `BoundKernel::resources()`. The decision
//! variable is the **partition** of the wave sequence into contiguous
//! groups. Each group becomes one `__global__` function on the target.
//!
//! For each candidate group `[j..i]`:
//! - **Hard constraint**: `max(wave.shmem_bytes) ≤
//!   LoweringConstraints::max_shmem_per_cta_bytes`. Violating this
//!   makes the group unschedulable on the target hardware.
//! - **Effective compute** = sum over waves in the group of
//!   `wave.compute_us × occupancy_penalty(wave, group_max_regs)`. The
//!   occupancy penalty captures the wall-clock cost of grouping a
//!   light-reg wave into a heavy-reg `__global__` (NVCC sets the per-
//!   `__global__` reg cap to `max(group)`, dropping occupancy for the
//!   lighter waves). When the target supports
//!   `regs_dynamic_per_warpgroup` (sm_90+ `setmaxnreg`), the penalty
//!   collapses to 1.0 and grouping is free.
//! - **In-group handoff cost** = `(group_size - 1) ×
//!   in_group_handoff_cost_us`. On sm_89 this is the grid sync cost
//!   (~100 µs). On sm_90+ it's the mbarrier shmem handoff (~1 µs).
//! - **Cross-group handoff** = `launch_cost_us` per group boundary
//!   (skipped for the very first group).
//!
//! ## DP recurrence
//!
//! `dp[i]` = minimum total wall-clock to lower waves `[0..i]`.
//!
//! ```text
//!   dp[0] = 0
//!   dp[i] = min over j in [0..i] of:
//!       dp[j]
//!     + group_cost(waves[j..i])
//!     + (j > 0 ? launch_cost_us : 0)
//! ```
//!
//! Time `O(N²)` in the wave count (~80 on Llama-1B → ~6400 iterations).
//! Backtrace from `dp[N]` to recover the optimal partition.
//!
//! ## Why DP is exact for this shape
//!
//! The wave sequence is linear (the BSP scheduler emits waves in
//! topological order, and each wave depends only on previous waves).
//! Group decisions don't have cross-cuts: a wave is in exactly one
//! group, the group is a contiguous range. The DP enumerates every
//! possible contiguous partition and picks the minimum-cost one.
//! Optimality is guaranteed.
//!
//! Future non-linear DAG cases (Step D cross-layer pipelining where
//! independent waves run on disjoint CTA subsets) need a different
//! solver — but for the current Llama prefill schedule the DAG is a
//! chain and DP is exact.

use std::collections::BTreeSet;

use crate::kernel_library::{CoalescedDag, Resources};
use crate::lowering::cost_occupancy::{blocks_per_sm_estimate, occupancy_penalty};
use crate::lowering::plan::{ExecutionPlan, Handoff, KernelGroup, WaveId};
use crate::schedule::{Wave, WaveSchedule};
use crate::target_profile::LoweringConstraints;

/// Per-wave summary the solver operates on. Built once from the
/// `WaveSchedule` + `CoalescedDag` before the DP loop runs.
#[derive(Clone, Debug)]
struct WaveSummary {
    /// Effective resource demand of this wave: max over the
    /// distinct bindings in the wave's CTA streams.
    resources: Resources,
    /// All distinct `BoundKernel::kind()` strings present in the
    /// wave. For monomorphic waves (the production case) this is a
    /// singleton.
    kinds: BTreeSet<&'static str>,
    /// Wave compute cost in microseconds (converted from the
    /// schedule's mma-unit `max_cta_cost` via
    /// `LoweringConstraints::gpu_clock_hz`).
    compute_us: f64,
}

fn summarize_waves(
    schedule: &WaveSchedule,
    dag: &CoalescedDag,
    constraints: &LoweringConstraints,
) -> Vec<WaveSummary> {
    schedule
        .waves
        .iter()
        .map(|w| summarize_wave(w, dag, constraints))
        .collect()
}

fn summarize_wave(
    wave: &Wave,
    dag: &CoalescedDag,
    constraints: &LoweringConstraints,
) -> WaveSummary {
    // Walk the wave's per-CTA streams; collect the distinct binding
    // refs by NodeId so we don't double-count wave-coop bindings
    // (which appear in every CTA's stream).
    let mut seen_node_ids: BTreeSet<u32> = BTreeSet::new();
    let mut max_shmem: u32 = 0;
    let mut max_regs: u32 = 0;
    let mut max_threads: u32 = 0;
    let mut kinds: BTreeSet<&'static str> = BTreeSet::new();
    for cta_stream in &wave.cta_nodes {
        for nid in cta_stream {
            if !seen_node_ids.insert(nid.0) {
                continue;
            }
            let kernel = &dag.nodes[nid.0 as usize].kernel;
            let r = kernel.resources();
            max_shmem = max_shmem.max(r.shmem_bytes);
            max_regs = max_regs.max(r.regs_per_thread);
            max_threads = max_threads.max(r.threads_per_cta);
            kinds.insert(kernel.kind());
        }
    }
    // Empty wave (shouldn't happen in production but be defensive).
    if max_threads == 0 {
        max_threads = 256;
    }
    let resources = Resources {
        shmem_bytes: max_shmem,
        regs_per_thread: max_regs.max(1),
        threads_per_cta: max_threads,
    };
    let compute_us = mma_units_to_us(wave.max_cta_cost, constraints);
    WaveSummary {
        resources,
        kinds,
        compute_us,
    }
}

/// Convert a `WaveSchedule` cost (in `mma units` ≈ cycles per the
/// existing `ms_at` convention in `schedule.rs`) to wall-clock
/// microseconds.
///
/// **Caveat**: the existing `CostModel` under-counts wall-clock by
/// ~20× at production dims (its mma-instruction estimates don't
/// account for cp.async pipeline fill, register pressure, occupancy,
/// etc.). The solver's decisions remain *qualitatively* correct
/// because it compares **per-wave compute** against absolute
/// `barrier_cost_us` / `launch_cost_us` from `LoweringConstraints` —
/// and those constants are wall-clock-calibrated. The dumped
/// `predicted_total_us` should be read as "the cost model's view of
/// the schedule", not "the wall-clock the lowered kernel will run in".
/// A future cost-model accuracy pass (Step beyond CP1) will close the
/// gap.
fn mma_units_to_us(mma_units: u64, c: &LoweringConstraints) -> f64 {
    (mma_units as f64) / c.gpu_clock_hz * 1_000_000.0
}

/// Cost of one candidate kernel group `[j..i]` in microseconds.
/// Returns `f64::INFINITY` if the group violates a hard constraint
/// (e.g. unioned shmem exceeds the per-CTA budget).
fn group_cost_us(group: &[WaveSummary], c: &LoweringConstraints) -> f64 {
    if group.is_empty() {
        return 0.0;
    }
    let group_max_shmem: u32 = group
        .iter()
        .map(|w| w.resources.shmem_bytes)
        .max()
        .unwrap_or(0);
    if group_max_shmem > c.max_shmem_per_cta_bytes {
        return f64::INFINITY;
    }
    let group_max_regs: u32 = group
        .iter()
        .map(|w| w.resources.regs_per_thread)
        .max()
        .unwrap_or(1);
    if group_max_regs > c.max_regs_per_thread {
        return f64::INFINITY;
    }
    let mut compute: f64 = 0.0;
    for wave in group {
        let pen = occupancy_penalty(wave.resources, group_max_regs, c) as f64;
        compute += wave.compute_us * pen;
    }
    let in_group_handoffs = group.len().saturating_sub(1) as f64;
    let in_group_handoff_cost = in_group_handoff_cost_us(c);
    compute + in_group_handoffs * in_group_handoff_cost
}

/// In-group handoff cost: mbarrier if the target has it (sm_90+),
/// otherwise the grid sync cost (sm_89).
fn in_group_handoff_cost_us(c: &LoweringConstraints) -> f64 {
    if let Some(mbarrier_us) = c.mbarrier_handoff_us {
        mbarrier_us as f64
    } else {
        c.barrier_cost_us as f64
    }
}

/// Pick the right `Handoff` variant for the boundary between two
/// groups on this target. The cost matches the constraint value the
/// solver used in its DP.
fn boundary_handoff(c: &LoweringConstraints) -> Handoff {
    Handoff::LaunchBoundary {
        cost_us: c.launch_cost_us,
    }
}

/// Pick the right in-group handoff variant for this target. Used by
/// the plan emit step (the solver itself just uses
/// `in_group_handoff_cost_us`).
fn in_group_handoff_kind(c: &LoweringConstraints) -> Handoff {
    if let Some(mbarrier_us) = c.mbarrier_handoff_us {
        Handoff::MbarrierShmem {
            cost_us: mbarrier_us,
        }
    } else {
        Handoff::InKernelGridSync {
            cost_us: c.barrier_cost_us,
        }
    }
}

/// Run the DP solver and produce the optimal `ExecutionPlan`.
///
/// **Inputs**:
/// - `schedule` — the BSP wave partition (output of `partition_into_waves`)
/// - `dag` — the coalesced DAG, used to look up `BoundKernel::resources()`
///   and `BoundKernel::kind()` for each wave
/// - `constraints` — target capability + cost numbers (read from
///   `TargetProfile::lowering`)
///
/// **Output**: an `ExecutionPlan` whose `groups` partition is the
/// minimum-cost lowering for the given target. Each group is a
/// contiguous slice of the input wave sequence.
pub fn lower(
    schedule: &WaveSchedule,
    dag: &CoalescedDag,
    constraints: &LoweringConstraints,
) -> ExecutionPlan {
    let waves = summarize_waves(schedule, dag, constraints);
    let n = waves.len();
    if n == 0 {
        return ExecutionPlan {
            groups: Vec::new(),
            predicted_total_us: 0.0,
            legacy_grid_sync_total_us: 0.0,
        };
    }

    // dp[i] = minimum cost to lower waves[0..i]. dp[0] = 0.
    // prev[i] = the j such that the optimal final group is waves[j..i].
    let mut dp = vec![f64::INFINITY; n + 1];
    let mut prev = vec![0usize; n + 1];
    dp[0] = 0.0;

    for i in 1..=n {
        for j in 0..i {
            let group_cost = group_cost_us(&waves[j..i], constraints);
            if !group_cost.is_finite() {
                continue;
            }
            let boundary_cost = if j > 0 {
                constraints.launch_cost_us as f64
            } else {
                0.0
            };
            let cand = dp[j] + group_cost + boundary_cost;
            if cand < dp[i] {
                dp[i] = cand;
                prev[i] = j;
            }
        }
    }

    // Backtrace from dp[n] to recover group boundaries.
    let mut boundaries: Vec<usize> = Vec::new();
    let mut i = n;
    while i > 0 {
        boundaries.push(i);
        i = prev[i];
    }
    boundaries.push(0);
    boundaries.reverse();
    // boundaries: [0, b1, b2, ..., n]; groups are waves[boundaries[k]..boundaries[k+1]].

    // Materialize the groups.
    let num_groups = boundaries.len() - 1;
    let mut groups: Vec<KernelGroup> = Vec::with_capacity(num_groups);
    for g_idx in 0..num_groups {
        let j = boundaries[g_idx];
        let i = boundaries[g_idx + 1];
        let group_waves = &waves[j..i];
        let group_max_shmem = group_waves
            .iter()
            .map(|w| w.resources.shmem_bytes)
            .max()
            .unwrap_or(0);
        let group_max_regs = group_waves
            .iter()
            .map(|w| w.resources.regs_per_thread)
            .max()
            .unwrap_or(1);
        let threads_per_cta = group_waves
            .iter()
            .map(|w| w.resources.threads_per_cta)
            .max()
            .unwrap_or(256);
        let mut kinds: BTreeSet<&'static str> = BTreeSet::new();
        for w in group_waves {
            for k in &w.kinds {
                kinds.insert(*k);
            }
        }
        let predicted_compute_us = group_cost_us(group_waves, constraints);
        let in_group_handoffs = (i - j).saturating_sub(1) as u32;
        let blocks_per_sm = blocks_per_sm_estimate(
            group_max_regs,
            threads_per_cta,
            group_max_shmem,
            constraints,
        );
        let handoff_to_next = if g_idx + 1 < num_groups {
            boundary_handoff(constraints)
        } else {
            Handoff::None
        };
        groups.push(KernelGroup {
            group_id: g_idx as u32,
            waves: (j..i).map(|k| k as WaveId).collect(),
            kinds,
            max_regs_per_thread: group_max_regs,
            max_shmem_per_cta: group_max_shmem,
            threads_per_cta,
            blocks_per_sm_estimate: blocks_per_sm,
            in_group_handoffs,
            handoff_to_next,
            predicted_compute_us,
        });
    }

    let predicted_total_us = dp[n];

    // Legacy grid_sync rollup: hypothetical "all waves in one __global__"
    // lowering would pay (n - 1) in-kernel grid syncs at the target's
    // barrier cost, plus the same effective compute (occupancy penalty
    // applied with group_max_regs = max over ALL waves' regs).
    let legacy_max_regs = waves
        .iter()
        .map(|w| w.resources.regs_per_thread)
        .max()
        .unwrap_or(1);
    let legacy_grid_sync_total_us = (n.saturating_sub(1) as f64)
        * (constraints.barrier_cost_us as f64)
        + waves
            .iter()
            .map(|w| {
                w.compute_us * occupancy_penalty(w.resources, legacy_max_regs, constraints) as f64
            })
            .sum::<f64>();

    let _ = in_group_handoff_kind(constraints); // reserved for sm_90 emit code

    ExecutionPlan {
        groups,
        predicted_total_us,
        legacy_grid_sync_total_us,
    }
}
