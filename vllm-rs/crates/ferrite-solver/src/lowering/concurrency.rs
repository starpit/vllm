// SPDX-License-Identifier: Apache-2.0
//! [`ConcurrencyModel`] — what overlaps on the target hardware.
//!
//! The solver consults this when computing the cost of two
//! implementations scheduled at the same time step. Without it the
//! solver would happily produce a multi-stream schedule that
//! doesn't actually run faster (e.g. two compute-bound GEMMs on
//! different streams that both hit the saturated tensor core).
//!
//! ## First-cut rules (sm_89, L4)
//!
//! Encoded as four hand-written rules:
//!
//! 1. **CooperativeLaunch is exclusive.** Only one cooperative grid
//!    can be resident on the device. Two CooperativeLaunch impls
//!    scheduled at the same step ⇒ infinite contention factor.
//!
//! 2. **Compute-bound impls don't overlap each other.** Two impls
//!    that are both `is_compute_bound() == true` and run on the
//!    same target's tensor cores get a contention factor of 1.0
//!    (no speedup from concurrency — they serialize on the
//!    saturated MMA pipeline).
//!
//! 3. **Memory-bound impls can overlap each other and a
//!    compute-bound impl.** A memory-bound op (norm, rope,
//!    silu_mul, residual_add) running concurrently with a
//!    compute-bound GEMM on a different stream actually overlaps
//!    on the target — the LSU and tensor cores are independent
//!    units. Contention factor for the memory-bound op is ~0.5
//!    (it runs roughly in the shadow of the compute-bound op).
//!
//! 4. **DeviceCallable inside a persistent megakernel** can
//!    overlap with other DeviceCallable impls in the same kernel
//!    if they're assigned to disjoint CTA subsets, modulo the
//!    resource-union constraint. The amortization factor is
//!    proportional to the CTA-subset coverage.
//!
//! These rules are minimal but enough to get the solver started.
//! Extension shape is documented inline.

use crate::lowering::implementation::{Implementation, LaunchKind};
use crate::target_profile::TargetProfile;

/// Multiplicative factor applied to an implementation's `cost_us`
/// when it runs concurrently with another. `1.0` means "no
/// speedup, no slowdown"; `< 1.0` means "this impl runs faster
/// because it overlaps with the other"; `f64::INFINITY` means
/// "this concurrent schedule is infeasible (e.g. two cooperative
/// grids)."
pub type ContentionFactor = f64;

/// First-cut concurrency model. Hand-written rules; extend as
/// measurements arrive.
pub struct ConcurrencyModel<'p> {
    pub profile: &'p TargetProfile,
}

impl<'p> ConcurrencyModel<'p> {
    pub fn new(profile: &'p TargetProfile) -> Self {
        Self { profile }
    }

    /// Contention factor for `target_impl` running concurrently
    /// with the set of `other_impls` already scheduled at the same
    /// time step.
    ///
    /// Returns `f64::INFINITY` if the schedule is infeasible (e.g.
    /// `target_impl` is `CooperativeLaunch` and the others include
    /// any other launch).
    pub fn contention_factor(
        &self,
        target_impl: &dyn Implementation,
        other_impls: &[&dyn Implementation],
    ) -> ContentionFactor {
        // Rule 1: CooperativeLaunch exclusivity.
        if matches!(target_impl.launch_kind(), LaunchKind::CooperativeLaunch)
            && !other_impls.is_empty()
        {
            return f64::INFINITY;
        }
        for other in other_impls {
            if matches!(other.launch_kind(), LaunchKind::CooperativeLaunch) {
                return f64::INFINITY;
            }
        }

        // Rule 2: compute-bound + compute-bound = no overlap.
        if target_impl.is_compute_bound() {
            let any_other_compute_bound = other_impls.iter().any(|o| o.is_compute_bound());
            if any_other_compute_bound {
                // Both compete for the tensor cores. No speedup.
                return 1.0;
            }
        }

        // Rule 3: memory-bound impl shadows behind a compute-bound
        // impl. The memory-bound impl gets a 0.5 contention factor
        // (effectively half the wall-clock cost) when there's a
        // compute-bound impl running concurrently. Compute-bound
        // impls are unaffected by memory-bound co-runners.
        if !target_impl.is_compute_bound() {
            let any_compute_bound = other_impls.iter().any(|o| o.is_compute_bound());
            if any_compute_bound {
                return 0.5;
            }
        }

        // Rule 4: DeviceCallable in same persistent kernel — not yet
        // exercised by the L4 sm_89 starter library, which is all
        // HostCallback. Stub for now; returns 1.0 (no concurrency
        // benefit by default; add the per-CTA-subset rule when the
        // first DeviceCallable+DeviceCallable case shows up).
        1.0
    }
}
