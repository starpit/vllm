// SPDX-License-Identifier: Apache-2.0
//! Interpreter backends for the universal `Instruction<W>` IR.
//!
//! Three sibling emitters consume the same lowered buckets that
//! `host::{colored_slot_map, apply_loop_compression,
//! collect_boundary_inputs}` produce. The post-solve selector
//! [`pick_interpreter`] decides which one runs based on (1) the
//! [`MegakernelFit`](crate::impl_lib::MegakernelFit) of every
//! solver-picked impl and (2) the target's capability flags. Both
//! axes must hold; otherwise the run falls through to `Host`.
//!
//! See `MEGA_HANDOFF.md` for the locked design.
//!
//! Today only `host` carries a body. `prim_mega` and `kvm_mega`
//! are skeletons; their encoder matches and launchers are filled
//! in as Phase 1 / Phase 2 of the work order land. The selector
//! always returns `Host` until eligibility wires up — its dead-
//! code allow pairs with the absence of a consumer; the call
//! site lands when codegen.rs starts emitting per-arch
//! interpreter selection.

#![allow(dead_code)]

pub mod host;
pub mod kvm_mega;
pub mod prim_mega;

use crate::impl_lib::{Implementation, MegakernelFit};
use crate::target::TargetProfile;

/// Which interpreter executes a canonical-bucket forward at run
/// time. Selected post-solve from the picked impl set + target
/// capability — the solver itself doesn't know about this split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpreter {
    /// Host loop over `Instruction<W>` slices. Today's only path.
    Host,
    /// Primitive megakernel — persistent `__global__` whose body
    /// is a switch over `[i32; 32]` opcodes calling `__device__`
    /// wrappers. Phase 1.
    PrimMega,
    /// KVM megakernel — vendored `~/Megakernels` template
    /// instantiated per arch with warp specialization. Phase 2.
    KvmMega,
}

/// Post-solve interpreter selection. The solver picks impls by
/// cost; this function picks the runtime that can execute them.
///
/// Tiering: a canonical is `KvmMega`-eligible only if every picked
/// impl is `Kvm`-fit AND the target supports KVM; falls back to
/// `PrimMega` if every pick is at least `Primitive`-fit AND the
/// target supports prim-mega; otherwise `Host`.
///
/// Today both target predicates return `false` (no DeviceCallable
/// impls exist yet), so this always returns `Host`. Wired through
/// so call sites land before the first `Primitive`-fit impl ships.
pub fn pick_interpreter(picked: &[&dyn Implementation], profile: &TargetProfile) -> Interpreter {
    let min_fit = picked
        .iter()
        .map(|i| i.megakernel_fit())
        .min()
        .unwrap_or(MegakernelFit::None);
    if profile.kvm_compatible() && min_fit >= MegakernelFit::Kvm {
        Interpreter::KvmMega
    } else if profile.prim_mega_compatible() && min_fit >= MegakernelFit::Primitive {
        Interpreter::PrimMega
    } else {
        Interpreter::Host
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_lib::{DcRmsNormImpl, RmsNormRefImpl};

    /// H100 / sm_90 — `prim_mega_compatible() == true`,
    /// `kvm_compatible() == false`. The two-axis selector should
    /// route through PrimMega when every picked impl is Primitive
    /// or better, and fall back to Host the moment a None-fit impl
    /// gets picked alongside. L4 stays Host even with a DC sibling
    /// — see `falls_back_to_host_on_ada_below_prim_mega_floor`.
    #[test]
    fn picks_prim_mega_when_all_impls_primitive_and_target_supports() {
        let profile = crate::target::from_profile_def(&ferrite_cuda_targets::H100_SM90);
        let dc = DcRmsNormImpl;
        let picked: Vec<&dyn Implementation> = vec![&dc];
        assert_eq!(pick_interpreter(&picked, &profile), Interpreter::PrimMega);
    }

    #[test]
    fn falls_back_to_host_when_any_pick_is_none_fit() {
        let profile = crate::target::from_profile_def(&ferrite_cuda_targets::H100_SM90);
        let dc = DcRmsNormImpl;
        let host = RmsNormRefImpl;
        // Mixed pick: host sibling drags `min_fit` down to None even
        // though the DC sibling alone would have qualified.
        let picked: Vec<&dyn Implementation> = vec![&dc, &host];
        assert_eq!(pick_interpreter(&picked, &profile), Interpreter::Host);
    }

    /// L4 (sm_89) is below the prim-mega floor — even if a DC
    /// sibling somehow ended up in the pick set, the target gate
    /// rejects PrimMega and we fall through to Host. `target_
    /// compatible()` on the DC siblings prevents the solver from
    /// picking them in the first place; this test pins the
    /// downstream selector behavior independently.
    #[test]
    fn falls_back_to_host_on_ada_below_prim_mega_floor() {
        let profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!profile.prim_mega_compatible());
        let dc = DcRmsNormImpl;
        let picked: Vec<&dyn Implementation> = vec![&dc];
        assert_eq!(pick_interpreter(&picked, &profile), Interpreter::Host);
    }

    #[test]
    fn empty_pick_set_falls_through_to_host() {
        // `min` over an empty iter returns None; the selector
        // defaults to None-fit and lands on Host. Pin that against
        // future refactors that might want to promote-on-empty.
        let profile = crate::target::from_profile_def(&ferrite_cuda_targets::H100_SM90);
        let picked: Vec<&dyn Implementation> = Vec::new();
        assert_eq!(pick_interpreter(&picked, &profile), Interpreter::Host);
    }
}
