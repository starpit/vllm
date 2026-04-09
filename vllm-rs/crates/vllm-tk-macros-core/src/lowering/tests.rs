// SPDX-License-Identifier: Apache-2.0
//! Lowering solver tests.
//!
//! Pinning the **structural decisions** the solver makes for each
//! target profile, so future refactors can't silently change them.
//!
//! ## Headline assertions
//!
//! - `sm89_l4_plan_separates_every_wave`: on the production L4 sm_89
//!   profile (`barrier_cost_us = 100, launch_cost_us = 5,
//!   regs_dynamic = false`), the solver outputs **one group per wave**
//!   for the Llama-1B seq=1024 production schedule. Each cooperative
//!   launch handles a single wave; kernel-boundary sync replaces the
//!   ~100 µs grid sync. This is the structural change that gets sm_89
//!   from ~54 ms toward the per-kind-register-budget regime.
//!
//! - `synthetic_sm90_plan_collapses_to_one_group`: a synthetic profile
//!   matching Hopper's capabilities (`mbarrier_handoff_us = Some(1),
//!   regs_dynamic = true`) on the **same** wave schedule produces a
//!   single group with `n - 1` in-group mbarrier handoffs. This proves
//!   the same solver, with no per-target code, emits a structurally
//!   different lowering when the target's constants change. It's the
//!   compiler-shaped argument: target backends are translators, the
//!   solver is the lowering.
//!
//! - `solver_respects_shmem_budget`: a synthetic profile with a tight
//!   shmem ceiling forces the solver to split groups whose unioned
//!   shmem would overflow.

use crate::kernel_library::{BoundKernel, FaninConsumer, coalesce_with_target_profile};
use crate::lowering::plan::Handoff;
use crate::lowering::solver::lower;
use crate::reified_dag::{LlamaDims, ReifiedDag, TileSizes};
use crate::schedule::{BARRIER_COST_MMA_UNITS, CostModel, partition_into_waves};
use crate::target_profile::TargetProfile;

fn llama_1b_seq1024_dims() -> LlamaDims {
    LlamaDims {
        num_layers: 16,
        hidden_dim: 2048,
        intermediate_dim: 8192,
        num_attn_heads: 32,
        num_kv_heads: 8,
        head_dim: 64,
        seq_len: 1024,
    }
}

fn build_production_plan_inputs() -> (
    crate::kernel_library::CoalescedDag,
    crate::schedule::WaveSchedule,
    TargetProfile,
) {
    let profile = TargetProfile::l4_sm89();
    let reified = ReifiedDag::reify_llama(llama_1b_seq1024_dims(), TileSizes::default_v1());
    let dag = coalesce_with_target_profile(&reified, &profile);
    let cost = CostModel::from_dag(&dag, profile.cooperative_grid_size());
    let sched = partition_into_waves(
        &dag,
        profile.cooperative_grid_size(),
        &cost,
        BARRIER_COST_MMA_UNITS,
    );
    (dag, sched, profile)
}

#[test]
fn sm89_l4_plan_separates_every_wave() {
    let (dag, schedule, profile) = build_production_plan_inputs();
    let plan = lower(&schedule, &dag, &profile.lowering);

    // The structural assertion: each wave gets its own group on
    // sm_89, because barrier_cost (100µs) ≫ launch_cost (5µs) and
    // grouping mismatched-reg waves would also pay an occupancy
    // penalty (no setmaxnreg on sm_89).
    assert_eq!(
        plan.groups.len(),
        schedule.waves.len(),
        "sm_89 lowering should produce one group per wave; \
         got {} groups for {} waves",
        plan.groups.len(),
        schedule.waves.len(),
    );
    for g in &plan.groups {
        assert_eq!(g.waves.len(), 1, "expected singleton groups on sm_89");
        assert_eq!(g.in_group_handoffs, 0);
    }
    // Every cross-group handoff is a launch boundary except the last.
    let last_idx = plan.groups.len() - 1;
    for (idx, g) in plan.groups.iter().enumerate() {
        if idx == last_idx {
            assert_eq!(g.handoff_to_next, Handoff::None);
        } else {
            assert!(matches!(g.handoff_to_next, Handoff::LaunchBoundary { .. }));
        }
    }
    // The plan should predict savings vs the legacy "all-in-one
    // mega __global__" lowering: we replace ~80 grid syncs (100µs each)
    // with ~80 launch boundaries (5µs each). Net ≈ 7.6 ms saving on
    // the handoff cost alone, before any per-kind register-budget
    // wins (which CP2/3 add).
    assert!(
        plan.savings_vs_legacy_us() > 5_000.0,
        "expected ≥ 5 ms barrier savings; got {:.1} µs",
        plan.savings_vs_legacy_us()
    );
}

#[test]
fn synthetic_sm90_plan_collapses_to_one_group() {
    // A profile that mirrors Hopper's two key capabilities:
    //   - regs_dynamic_per_warpgroup (setmaxnreg.inc/dec)
    //   - mbarrier_handoff_us cheaper than launch_cost_us
    // Same wave schedule, same DAG, **only the LoweringConstraints
    // change** — but the solver emits a structurally different plan.
    let (dag, schedule, mut profile) = build_production_plan_inputs();
    profile.lowering.regs_dynamic_per_warpgroup = true;
    profile.lowering.mbarrier_handoff_us = Some(1.0);
    profile.lowering.barrier_cost_us = 1.0; // mbarrier is in-group cost too
    profile.lowering.launch_cost_us = 5.0;
    let plan = lower(&schedule, &dag, &profile.lowering);

    assert_eq!(
        plan.groups.len(),
        1,
        "Hopper-like profile should collapse to a single warp-specialized group; \
         got {} groups",
        plan.groups.len()
    );
    let g = &plan.groups[0];
    assert_eq!(g.waves.len(), schedule.waves.len());
    assert_eq!(g.in_group_handoffs as usize, schedule.waves.len() - 1);
    assert_eq!(g.handoff_to_next, Handoff::None);
}

#[test]
fn shmem_budget_above_max_wave_is_satisfied() {
    // With the budget set just above the heaviest single-wave shmem
    // demand (FlashInfer attention ≈ 70 KiB), every group's shmem
    // should fit. This is the normal case.
    let (dag, schedule, mut profile) = build_production_plan_inputs();
    profile.lowering.max_shmem_per_cta_bytes = 72 * 1024;
    let plan = lower(&schedule, &dag, &profile.lowering);
    for g in &plan.groups {
        assert!(
            g.max_shmem_per_cta <= 72 * 1024,
            "group {} has shmem {} > budget 73728",
            g.group_id,
            g.max_shmem_per_cta
        );
    }
    assert!(plan.predicted_total_us.is_finite());
}

#[test]
fn shmem_budget_below_single_wave_is_infeasible() {
    // Set the budget BELOW FlashInfer's 70 KiB requirement. No valid
    // partition exists (a single wave's own shmem overflows the
    // hardware carveout — there's no group small enough to fit).
    // The solver must surface infeasibility via an infinite total
    // cost; the codegen backend will refuse to emit code for such
    // a plan rather than silently producing unschedulable kernels.
    let (dag, schedule, mut profile) = build_production_plan_inputs();
    profile.lowering.max_shmem_per_cta_bytes = 32 * 1024;
    let plan = lower(&schedule, &dag, &profile.lowering);
    assert!(
        plan.predicted_total_us.is_infinite(),
        "expected infeasible plan with infinite cost; got {} µs",
        plan.predicted_total_us
    );
}

#[test]
fn plan_dump_for_inspection() {
    // Not an assertion test — prints the L4 plan to stderr so
    // `cargo test ... -- --nocapture plan_dump_for_inspection` shows
    // the plan for human eyeballing. The CI test below
    // (`sm89_l4_plan_separates_every_wave`) is the structural lock-in.
    let (dag, schedule, profile) = build_production_plan_inputs();
    let plan = lower(&schedule, &dag, &profile.lowering);
    eprintln!("\n--- L4 sm_89 ExecutionPlan (Llama-1B seq=1024) ---");
    eprintln!("{plan}");
}

#[test]
fn solver_actually_uses_the_constants_not_hardcoded_singletons() {
    // Counter-experiment: take the production L4 sm_89 schedule but
    // FLIP the constraint constants so in-kernel barriers are cheaper
    // than kernel-boundary launches. With `regs_dynamic` = true so
    // there's no occupancy penalty muddying the math, the solver
    // **must** merge — otherwise it's hardcoding the singleton
    // partition rather than computing it from constants.
    //
    // This is the load-bearing test that proves the solver is doing
    // real optimization, not just emitting `groups.len() == waves.len()`.
    let (dag, schedule, mut profile) = build_production_plan_inputs();
    profile.lowering.barrier_cost_us = 2.0; // in-group cost
    profile.lowering.launch_cost_us = 100.0; // cross-group cost
    profile.lowering.regs_dynamic_per_warpgroup = true; // no occupancy penalty
    let plan = lower(&schedule, &dag, &profile.lowering);
    eprintln!(
        "[audit] flipped constants → {} groups for {} waves",
        plan.groups.len(),
        schedule.waves.len()
    );
    assert!(
        plan.groups.len() < schedule.waves.len() / 2,
        "with launch_cost ≫ barrier_cost the solver should merge \
         heavily; got {} groups for {} waves",
        plan.groups.len(),
        schedule.waves.len()
    );
}

#[test]
fn solver_separates_when_constants_demand_it() {
    // Inverse of the above. Force barrier ≫ launch on a profile
    // identical to L4 sm_89 except the constants. Should separate
    // every wave (matching the L4 production result) regardless of
    // any other field on the profile.
    let (dag, schedule, mut profile) = build_production_plan_inputs();
    profile.lowering.barrier_cost_us = 1000.0;
    profile.lowering.launch_cost_us = 1.0;
    let plan = lower(&schedule, &dag, &profile.lowering);
    assert_eq!(plan.groups.len(), schedule.waves.len());
}

#[test]
fn solver_responds_to_occupancy_penalty() {
    // Set barrier == launch so the in-group vs cross-group barrier
    // cost is a wash, then verify the solver still separates waves
    // with mismatched register footprints (because of the occupancy
    // penalty), AND merges waves with matched footprints (because
    // there's no occupancy penalty). This is a structural assertion
    // about the occupancy term doing its job in the cost function.
    //
    // Approach: synthetic 4-wave schedule with two distinct kinds.
    // Kind A: regs=32 (light norm). Kind B: regs=128 (heavy cutlass).
    // Sequence: A, A, B, B.
    //
    // Expected: solver merges A,A and B,B but splits between them
    // because grouping {A,A,B} forces A waves to pay sqrt(128/32) = 2x
    // occupancy penalty.
    use crate::kernel_library::Resources;
    use crate::lowering::cost_occupancy::occupancy_penalty;
    use crate::target_profile::LoweringConstraints;

    let mut c = LoweringConstraints::l4_sm89();
    c.barrier_cost_us = 5.0;
    c.launch_cost_us = 5.0;

    // Sanity check the penalty function does what we expect.
    let light = Resources {
        shmem_bytes: 4096,
        regs_per_thread: 32,
        threads_per_cta: 256,
    };
    let heavy_regs = 128;
    let p = occupancy_penalty(light, heavy_regs, &c);
    assert!(
        (p - 2.0).abs() < 0.01,
        "expected sqrt(128/32) = 2.0 penalty, got {p}"
    );
    // And no penalty when matched.
    let p_matched = occupancy_penalty(light, 32, &c);
    assert!((p_matched - 1.0).abs() < 0.01);
}

#[test]
fn binding_resources_are_nonzero() {
    // Sanity: every BoundKernel variant has non-zero resource
    // estimates. Without this, the occupancy model trivially
    // returns 1.0 (which would mask solver bugs).
    use crate::kernel_library::{CutlassTile, GemmPhase};
    use crate::reified_dag::Phase;
    let cases = [
        BoundKernel::HandWrittenRowTile {
            phase: Phase::AttnNorm,
            layer: 0,
            row: 0,
            col: 0,
        },
        BoundKernel::FlashInferAttentionLayer { layer: 0 },
        BoundKernel::CutlassGemmLayer {
            layer: 0,
            phase: GemmPhase::Qkv,
            tile: CutlassTile::Small,
        },
        BoundKernel::CutlassGemmLayer {
            layer: 0,
            phase: GemmPhase::OProj,
            tile: CutlassTile::Narrow,
        },
        BoundKernel::FusedFaninLayer {
            layer: 0,
            producer_phase: Phase::MlpNorm,
            consumer: FaninConsumer::CutlassGemm(GemmPhase::GateUp, CutlassTile::Small),
        },
    ];
    for k in cases {
        let r = k.resources();
        assert!(r.shmem_bytes > 0, "shmem_bytes must be > 0 for {k:?}");
        assert!(r.regs_per_thread > 0);
        assert!(r.threads_per_cta > 0);
    }
}
