// SPDX-License-Identifier: Apache-2.0
//! Occupancy / register-pressure modeling for the lowering solver.
//!
//! When the solver groups two waves into the same `__global__`, NVCC
//! sets the per-`__global__` register cap to the **maximum** of the
//! waves' `regs_per_thread` (unless the target supports
//! `setmaxnreg.inc/dec`, in which case each warpgroup picks its own
//! at runtime). The "lighter" waves in the group then run at the
//! grouped register count → fewer warps resident per SM → less
//! latency hiding for memory-bound segments → wall-clock slowdown.
//!
//! This file gives the solver a (deliberately simple) formula to
//! quantify that slowdown so the DP cost function can compare it
//! against the saving from replacing a launch boundary with an
//! in-kernel grid sync.
//!
//! **Why a simple formula**: the exact slowdown is hardware-specific,
//! workload-specific, and notoriously hard to predict from first
//! principles. The solver's job is to make the **right qualitative
//! decision** ("merge waves with similar reg footprints, separate
//! waves with very different ones"), not to predict bench numbers.
//! A square-root model captures the diminishing-returns shape of
//! occupancy → latency-hiding without overfitting.
//!
//! **Calibration**: future work, once CP2/3 produces measured per-kind
//! kernel timings on L4. The current formula is a placeholder; the
//! solver structure doesn't depend on its exact constants.

use crate::kernel_library::Resources;
use crate::target_profile::LoweringConstraints;

/// Multiplicative wall-clock penalty applied to a wave's compute time
/// when its enclosing kernel group has a higher per-thread register
/// cap than the wave itself needs.
///
/// Returns ≥ 1.0; equals 1.0 when the cap matches the wave's needs
/// or the target supports `setmaxnreg.inc/dec` (in which case each
/// warpgroup picks its own register count at runtime, so grouping
/// is free).
///
/// Formula (initial, will refine after CP2 measurement data):
///
/// ```text
///   penalty = max(1.0, sqrt(group_max_regs / wave_regs))
/// ```
///
/// This says "doubling the register cap on a wave that didn't need
/// it costs ~1.4× wall-clock". For the L4 numbers in
/// `BoundKernel::resources()`, grouping a `regs=24` norm wave with a
/// `regs=128` cutlass wave gives a penalty of √(128/24) ≈ 2.31× on
/// the norm wave's already-tiny compute, and 1.0× on the cutlass
/// wave (matched). The solver will (correctly) prefer to put the
/// norm wave in its own group on sm_89, where the launch overhead
/// is much smaller than the saved penalty × cutlass-grid-sync cost.
pub fn occupancy_penalty(
    wave_resources: Resources,
    group_max_regs_per_thread: u32,
    constraints: &LoweringConstraints,
) -> f32 {
    if constraints.regs_dynamic_per_warpgroup {
        // setmaxnreg lets each warpgroup pick its own reg count at
        // runtime. No penalty for grouping waves with mismatched
        // reg footprints.
        return 1.0;
    }
    if group_max_regs_per_thread <= wave_resources.regs_per_thread {
        return 1.0;
    }
    let ratio = group_max_regs_per_thread as f32 / wave_resources.regs_per_thread.max(1) as f32;
    ratio.sqrt().max(1.0)
}

/// Estimated resident blocks per SM for a kernel group with the
/// given per-thread register count and per-CTA shmem footprint.
/// Used by the plan dump for human inspection — not consumed by the
/// solver itself.
///
/// Computed as `min(regs_per_sm / (regs_per_thread * threads_per_cta),
/// shmem_per_sm / shmem_per_cta, warps_per_sm / (threads_per_cta/32))`
/// then clamped to ≥ 1 (at least one block must fit).
pub fn blocks_per_sm_estimate(
    regs_per_thread: u32,
    threads_per_cta: u32,
    shmem_per_cta: u32,
    constraints: &LoweringConstraints,
) -> u32 {
    let warps_per_cta = threads_per_cta.div_ceil(32);
    let by_warps = constraints.warps_per_sm / warps_per_cta.max(1);
    let by_regs = constraints
        .regs_per_sm
        .checked_div(regs_per_thread.max(1) * threads_per_cta.max(1))
        .unwrap_or(1);
    // Approximation: assume the SM's max dynamic shmem carveout
    // equals `max_shmem_per_cta_bytes`. Real SMs have a separate
    // total shmem budget but for the L4 99 KiB carveout the practical
    // ceiling is "1 block at the carveout, more if the block uses
    // less". We model this as floor(max / shmem) blocks per SM.
    let by_shmem = if shmem_per_cta == 0 {
        constraints.warps_per_sm
    } else {
        constraints.max_shmem_per_cta_bytes / shmem_per_cta
    };
    by_warps.min(by_regs).min(by_shmem).max(1)
}
