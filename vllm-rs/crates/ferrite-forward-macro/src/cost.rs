// SPDX-License-Identifier: Apache-2.0
//! Post-scheduler cost aggregation with per-wave contention.
//!
//! The solver's DP sums per-Impl costs independently — fine for
//! picking decisions (claim-size DESC breaks ties; cost ASC breaks
//! within size). But the reported wall-clock number should reflect
//! what actually happens on the device: Impls in the same wave
//! contend or overlap according to [`ConcurrencyModel`].
//!
//! This module walks the [`Loop`] per workload point, groups
//! Impls by wave, applies `contention_factor`, and returns the
//! adjusted total µs.
//!
//! Ported in spirit from `ferrite-solver/src/lowering/cost.rs`.
//! Per-launch sync overhead lands here too — every Impl pays one
//! [`launch_overhead_us`](crate::impl_lib::launch_overhead_us)
//! charge per pick, derived from its `launch_kind()`. The DP at
//! `solver.rs` adds the same term to candidate costs, so this
//! aggregator agrees with what was optimized. Per-edge handoff
//! costs (StreamEvent across compute units, GmemFlag spin, etc.)
//! still aren't modelled — they only matter once the schedule
//! routes onto multiple streams or splits a megakernel-internal
//! op chain across compilation units.

#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::concurrency::ConcurrencyModel;
use crate::fuf::Fuf;
use crate::impl_lib::{
    CostCtx, Implementation, ImplementationLibrary, MatchInfo, launch_overhead_us,
};
use crate::schedule::{Loop, WorkloadLoops};
use crate::solver::{Assignment, WorkloadAssignments};
use crate::target::TargetProfile;

/// Contention-aware total µs for a single workload bucket's LOOP.
///
/// Walks each wave, computes each Impl's cost with the other Impls
/// in the same wave as concurrent co-runners, sums. Returns
/// `f64::INFINITY` if any wave contains an infeasible combination
/// (two cooperative grids).
pub fn loop_cost_us(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    lib: &ImplementationLibrary,
    target: &TargetProfile,
    bounds: &BTreeMap<String, u64>,
) -> f64 {
    let concurrency = ConcurrencyModel::new(target);
    let ctx = CostCtx {
        fuf,
        profile: target,
        bounds,
    };

    let mut total = 0.0_f64;

    for wave in &loop_ir.waves {
        // Resolve each subgraph in this wave to (Impl, MatchInfo).
        let mut wave_entries: Vec<(&dyn Implementation, MatchInfo)> =
            Vec::with_capacity(wave.subgraphs.len());
        for (sg, imp_id) in &wave.subgraphs {
            let claimed = sfuf.tiles_in_subgraph(*sg);
            let m = MatchInfo {
                claimed_tiles: claimed,
                boundary_inputs: vec![],
                boundary_outputs: vec![],
            };
            wave_entries.push((lib.get(*imp_id), m));
        }

        // For each entry, apply contention_factor(self, others_in_wave).
        for i in 0..wave_entries.len() {
            let (imp, m) = &wave_entries[i];
            let others: Vec<&dyn Implementation> = wave_entries
                .iter()
                .enumerate()
                .filter_map(|(j, (oi, _))| if j == i { None } else { Some(*oi) })
                .collect();
            let factor = concurrency.contention_factor(*imp, &others);
            if !factor.is_finite() {
                return f64::INFINITY;
            }
            // Effective cost = GPU work × contention + per-launch
            // sync overhead. Contention only scales the work term —
            // running two kernels concurrently doesn't halve their
            // launch boundary; each still pays its own. Mirrors the
            // solver's phase-1 candidate cost so `predicted_us`
            // agrees with what the DP optimized.
            total += imp.cost_us(m, &ctx) * factor + launch_overhead_us(imp.launch_kind(), target);
        }
    }

    total
}

/// Overwrite each `Assignment.predicted_us` with the
/// contention-aware aggregate over its paired `Loop`. Called from
/// the macro drive after solve + schedule.
pub fn refresh_predicted_us(
    fuf: &Fuf,
    sfufs: &mut WorkloadAssignments,
    loops: &WorkloadLoops,
    lib: &ImplementationLibrary,
    target: &TargetProfile,
    bounds: &BTreeMap<String, u64>,
) {
    let mut scratch = bounds.clone();
    for (wp, sfuf) in sfufs.per_workload.iter_mut() {
        scratch.insert("num_tokens".into(), wp.num_tokens);
        scratch.insert("sk_bucket".into(), wp.sk_bucket);
        let Some(loop_ir) = loops.per_workload.get(wp) else {
            continue;
        };
        sfuf.predicted_us = loop_cost_us(fuf, sfuf, loop_ir, lib, target, &scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::build_cfg;
    use crate::classify::classify;
    use crate::config::{self, ModelParams};
    use crate::fuf::unroll;
    use crate::impl_lib::starter_library;
    use crate::parse::parse_block;
    use crate::schedule::schedule_workloads;
    use crate::shape::infer;
    use crate::solver::solve;
    use crate::target::from_profile_def;
    use std::path::PathBuf;

    fn llama_params(stem: &str) -> ModelParams {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("ferrite-model-llama")
            .join("configs")
            .join(format!("{stem}.json"));
        config::load_file(&path).unwrap()
    }

    fn l4_target() -> TargetProfile {
        from_profile_def(&ferrite_cuda_targets::L4_SM89)
    }

    fn h100_target() -> TargetProfile {
        from_profile_def(&ferrite_cuda_targets::H100_SM90)
    }

    const LLAMA_BODY: &str = r#"
        hidden_states = embed(input_ids, embed_tokens);
        for layer in 0..num_hidden_layers {
            normed = rmsnorm(hidden_states, input_layernorm[layer]);
            q = gemm(normed, self_attn.q_proj[layer]);
            k = gemm(normed, self_attn.k_proj[layer]);
            v = gemm(normed, self_attn.v_proj[layer]);
            (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
            attn = attention(q, k, v, kv_cache[layer], block_table);
            oproj = gemm(attn, self_attn.o_proj[layer]);
            hidden_states = add(oproj, hidden_states);

            normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
            gate = silu(gemm(normed2, mlp.gate_proj[layer]));
            up = gemm(normed2, mlp.up_proj[layer]);
            down = gemm(gate * up, mlp.down_proj[layer]);
            hidden_states = add(down, hidden_states);
        }
        normed = rmsnorm(hidden_states, norm);
        logits = gemm(normed, lm_head);
    "#;

    #[test]
    fn loop_cost_equals_naive_sum_on_serial_chain() {
        // Post-fusion Llama is a pure serial chain: every wave has
        // exactly one subgraph, so contention factor is always 1.0.
        // Consequence: contention-aware total matches the DP's naive
        // per-Impl sum. If this test fails, either the scheduler
        // started grouping independent subgraphs into waves (good —
        // update the test to match) or the concurrency model's
        // factor-1 identity broke.
        let params = llama_params("llama-3.2-1b");
        let file: syn::File =
            syn::parse_str(&format!("fn _c() {{ {LLAMA_BODY} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).unwrap();
        let program = classify(&ast).unwrap();
        let inferred = infer(
            &program,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &std::collections::BTreeMap::new(),
        )
        .unwrap();
        let cfg = build_cfg(&program, &params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        let lib = starter_library();
        let target = l4_target();

        let sfufs = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1], &[]).unwrap();
        let sfuf = sfufs.get_nt(1).unwrap();
        let loops = schedule_workloads(&fuf, &sfufs);
        let wp1 = crate::solver::WorkloadPoint::num_tokens_only(1);
        let loop_ir = &loops.per_workload[&wp1];

        let mut scratch = params.bounds.clone();
        scratch.insert("num_tokens".into(), 1);

        let naive = sfuf.predicted_us;
        let aware = loop_cost_us(&fuf, sfuf, loop_ir, &lib, &target, &scratch);

        // Every wave has one subgraph → contention factor 1.0
        // everywhere. Numbers must match to within float-noise
        // (both sums include the same Impl cost_us calls).
        assert!(
            (naive - aware).abs() < 1e-6,
            "naive={naive} aware={aware} (expected equal on serial chain)"
        );
    }

    /// `loop_cost_us` must be deterministic across repeated calls on
    /// the same inputs. Non-determinism here (old ferrite-solver had
    /// a HashMap-iteration bug that caused this) would invalidate
    /// any future cost-aware optimization pass: two calls in a row
    /// on the same assignment returned different numbers.
    #[test]
    fn loop_cost_us_is_deterministic() {
        let params = llama_params("llama-3.2-1b");
        let file: syn::File =
            syn::parse_str(&format!("fn _c() {{ {LLAMA_BODY} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).unwrap();
        let program = classify(&ast).unwrap();
        let inferred = infer(
            &program,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &std::collections::BTreeMap::new(),
        )
        .unwrap();
        let cfg = build_cfg(&program, &params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        let lib = starter_library();
        let target = l4_target();

        let sfufs = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512],
            &[],
        )
        .unwrap();
        let loops = schedule_workloads(&fuf, &sfufs);

        let mut scratch = params.bounds.clone();
        for &m in &[1u64, 512] {
            scratch.insert("num_tokens".into(), m);
            let wp = crate::solver::WorkloadPoint::num_tokens_only(m);
            let sfuf = sfufs.get_nt(m).unwrap();
            let loop_ir = &loops.per_workload[&wp];
            let first = loop_cost_us(&fuf, sfuf, loop_ir, &lib, &target, &scratch);
            for _ in 0..20 {
                let again = loop_cost_us(&fuf, sfuf, loop_ir, &lib, &target, &scratch);
                assert!(
                    (first - again).abs() < 1e-9,
                    "loop_cost_us returned {first} then {again} at num_tokens={m}"
                );
            }
        }
    }

    #[test]
    fn refresh_predicted_us_overwrites_in_place() {
        let params = llama_params("llama-3.2-1b");
        let file: syn::File =
            syn::parse_str(&format!("fn _c() {{ {LLAMA_BODY} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).unwrap();
        let program = classify(&ast).unwrap();
        let inferred = infer(
            &program,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &std::collections::BTreeMap::new(),
        )
        .unwrap();
        let cfg = build_cfg(&program, &params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        let lib = starter_library();
        let target = l4_target();

        let mut sfufs = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512],
            &[],
        )
        .unwrap();
        let before: BTreeMap<crate::solver::WorkloadPoint, f64> = sfufs
            .per_workload
            .iter()
            .map(|(wp, sf)| (*wp, sf.predicted_us))
            .collect();
        let loops = schedule_workloads(&fuf, &sfufs);
        refresh_predicted_us(&fuf, &mut sfufs, &loops, &lib, &target, &params.bounds);
        for (wp, sf) in sfufs.per_workload.iter() {
            assert!(sf.predicted_us.is_finite() && sf.predicted_us > 0.0);
            // On serial chain the refresh shouldn't move the number.
            assert!((before[wp] - sf.predicted_us).abs() < 1e-6);
        }
    }

    /// Lock the launch-overhead constants. The DP and `loop_cost_us`
    /// both consult `launch_overhead_us`; pin the values so an
    /// unrelated edit can't silently re-rank host vs DC at the
    /// solver.
    #[test]
    fn launch_overhead_matches_handoff_table() {
        use crate::impl_lib::{Handoff, LaunchKind, launch_overhead_us};
        for target in [l4_target(), h100_target()] {
            // Host launches all collapse to KernelBoundary — one
            // cudaLaunchKernel boundary regardless of whether the
            // host is calling cuBLAS or its own __global__. Same
            // value on every target (KernelBoundary is target-
            // agnostic in `Handoff::cost_us`).
            for lk in [
                LaunchKind::HostCallback,
                LaunchKind::RegularLaunch,
                LaunchKind::CooperativeLaunch,
            ] {
                assert_eq!(
                    launch_overhead_us(lk, &target),
                    Handoff::KernelBoundary.cost_us(&target),
                    "{lk:?} should pay KernelBoundary on {}",
                    target.name,
                );
            }
            // DC pays zero per pick — mega launch overhead is a
            // wave-level term, amortized over the whole DC chain.
            // Charging `InKernelGridSync` per pick double-counts
            // (vs the future wave-level launch counter) AND made
            // the DP never pick DC siblings, which directly
            // contradicted the older `worktree-ferrite-mega`
            // branch's shipped behavior on H100.
            assert_eq!(
                launch_overhead_us(LaunchKind::DeviceCallable, &target),
                0.0,
                "DC must pay zero per-pick on {} — wave-level term covers it",
                target.name,
            );
            // Solver effect: DC strictly cheaper than host at every
            // seed where DC is feasible. Feasibility is gated by
            // `prim_mega_compatible()` (sm_90+), so on Ada this
            // ranking never matters in practice; on Hopper it's
            // what flips the picks.
            assert!(
                launch_overhead_us(LaunchKind::DeviceCallable, &target)
                    < launch_overhead_us(LaunchKind::HostCallback, &target),
                "DC overhead must be strictly below host overhead on {} \
                 — otherwise the solver never picks DC and prim-mega is \
                 dead even on Hopper",
                target.name,
            );
        }
    }

    /// Solver `predicted_us` should now reflect per-Impl launch
    /// overhead. On a serial-chain Llama at M=1 the DP picks ~N
    /// host kernels (roughly one per FUF subgraph after fusion);
    /// the launch surcharge is `N × 5us`, which is observable.
    #[test]
    fn predicted_us_includes_launch_overhead() {
        use crate::impl_lib::{Handoff, LaunchKind, launch_overhead_us};
        let params = llama_params("llama-3.2-1b");
        let file: syn::File =
            syn::parse_str(&format!("fn _c() {{ {LLAMA_BODY} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).unwrap();
        let program = classify(&ast).unwrap();
        let inferred = infer(
            &program,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &std::collections::BTreeMap::new(),
        )
        .unwrap();
        let cfg = build_cfg(&program, &params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        let lib = starter_library();
        let target = l4_target();

        let sfufs = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1], &[]).unwrap();
        let sfuf = sfufs.get_nt(1).unwrap();

        // Compute the surcharge attributable to launch overhead by
        // summing it over picked impls. Subtract from predicted_us:
        // what's left is pure GPU work, must still be positive.
        let overhead: f64 = sfuf
            .impls
            .values()
            .map(|imp_id| launch_overhead_us(lib.get(*imp_id).launch_kind(), &target))
            .sum();
        assert!(
            overhead > 0.0,
            "Llama solve must pick at least one impl with non-zero overhead"
        );
        // Every host-launched impl contributes 5us; chain has
        // tens of subgraphs after fusion; surcharge dwarfs the
        // float-noise threshold by orders of magnitude.
        let host_floor = Handoff::KernelBoundary.cost_us(&target) * 10.0;
        assert!(
            overhead >= host_floor,
            "expected ≥{host_floor}us launch surcharge across the chain, got {overhead}"
        );
        let work = sfuf.predicted_us - overhead;
        assert!(
            work.is_finite() && work > 0.0,
            "predicted_us={} overhead={} → work={} should be positive",
            sfuf.predicted_us,
            overhead,
            work,
        );
        // L4 (Ada) is below `prim_mega_compatible()` — DC siblings
        // are filtered out at `target_compatible()` time, so the
        // DP can't pick any. Pin that against future regressions
        // that might relax the gate without re-checking the cost
        // model.
        for imp_id in sfuf.impls.values() {
            assert!(
                !matches!(lib.get(*imp_id).launch_kind(), LaunchKind::DeviceCallable),
                "DP picked a DC sibling at M=1 on L4 unexpectedly: \
                 {} — DC siblings should be filtered out by \
                 `target_compatible()` on Ada",
                lib.get(*imp_id).name(),
            );
        }
    }

    /// On Hopper the DC-sibling gate opens, host pays +5us per
    /// pick, DC pays +0us — DC strictly cheaper at every seed
    /// where it's solver-feasible. The DP must pick DC for at
    /// least the rms_norm seeds. Counterpart of
    /// `predicted_us_includes_launch_overhead` (which pins the L4
    /// half), this test pins the H100 half: the cost-model split
    /// I plumbed in (`launch_overhead_us`,
    /// `prim_mega_compatible() >= 90`, target-aware
    /// `Handoff::InKernelGridSync`) flips picks correctly.
    #[test]
    fn solver_picks_dc_siblings_on_h100() {
        use crate::impl_lib::LaunchKind;
        let params = llama_params("llama-3.2-1b");
        let file: syn::File =
            syn::parse_str(&format!("fn _c() {{ {LLAMA_BODY} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).unwrap();
        let program = classify(&ast).unwrap();
        let inferred = infer(
            &program,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &std::collections::BTreeMap::new(),
        )
        .unwrap();
        let cfg = build_cfg(&program, &params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        let lib = starter_library();
        let target = h100_target();

        let sfufs = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1], &[]).unwrap();
        let sfuf = sfufs.get_nt(1).unwrap();

        let dc_picks: Vec<&'static str> = sfuf
            .impls
            .values()
            .filter_map(|imp_id| {
                let imp = lib.get(*imp_id);
                if matches!(imp.launch_kind(), LaunchKind::DeviceCallable) {
                    Some(imp.name())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            !dc_picks.is_empty(),
            "expected at least one DC sibling picked on H100; got zero. \
             `launch_overhead_us(DC, h100) == 0` and \
             `launch_overhead_us(Host, h100) == 5us`, so DC should win \
             every per-seed cost comparison where it's feasible. Either \
             `prim_mega_compatible()` regressed below sm_90, the DC \
             siblings' `target_compatible()` got tightened, or someone \
             un-zeroed DC's per-pick overhead in `launch_overhead_us`."
        );
        // Sanity-check: at least DcRmsNormImpl should be picked
        // since RmsNorm is the simplest seed in the body and DC's
        // claim shape matches RmsNormRefImpl exactly via thin
        // delegation.
        assert!(
            dc_picks.iter().any(|n| n.contains("rmsnorm")),
            "expected `dc_rmsnorm` among DC picks on H100, got: {dc_picks:?}"
        );
    }
}
