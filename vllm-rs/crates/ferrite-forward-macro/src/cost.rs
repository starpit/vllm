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
//! Ported in spirit from `ferrite-solver/src/lowering/cost.rs` —
//! simplified because our assignment doesn't yet carry
//! `CompilationUnitId`/Handoff edges. Those pieces land with their
//! consumers (DeviceCallable impls + megakernel emission).

#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::concurrency::ConcurrencyModel;
use crate::fuf::Fuf;
use crate::impl_lib::{CostCtx, Implementation, ImplementationLibrary, LaunchKind, MatchInfo};
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

        if wave.is_megakernel {
            // Megakernel phases run in sequence with grid sync —
            // no contention, factor 1.0 for each phase.
            for (imp, m) in &wave_entries {
                total += imp.cost_us(m, &ctx);
            }
        } else {
            // Parallel wave: apply contention_factor for co-runners.
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
                total += imp.cost_us(m, &ctx) * factor;
            }
        }
    }

    // Add per-launch overhead. CSV costs are compute-only (launch
    // overhead subtracted from measured timings).
    //   - Each HostCallback / RegularLaunch subgraph pays one launch.
    //   - A megakernel wave pays one cooperative launch for all its
    //     DC subgraphs (that's the whole point of megakerneling).
    //   - Standalone DeviceCallable subgraphs in non-megakernel waves
    //     pay nothing (they run inside an enclosing kernel).
    let mut launch_count = 0u32;
    for wave in &loop_ir.waves {
        if wave.is_megakernel {
            // One cooperative launch for the whole wave.
            launch_count += 1;
        } else {
            for (_sg, imp_id) in &wave.subgraphs {
                match lib.get(*imp_id).launch_kind() {
                    LaunchKind::HostCallback | LaunchKind::RegularLaunch => {
                        launch_count += 1;
                    }
                    LaunchKind::CooperativeLaunch | LaunchKind::DeviceCallable => {}
                }
            }
        }
    }
    total += launch_count as f64 * target.launch_overhead_us;

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
    for (&m, sfuf) in sfufs.per_num_tokens.iter_mut() {
        scratch.insert("num_tokens".into(), m);
        let Some(loop_ir) = loops.per_num_tokens.get(&m) else {
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
    use crate::target::load_file as load_target;
    use std::path::PathBuf;

    fn llama_params(stem: &str) -> ModelParams {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("model_architectures")
            .join("llama")
            .join(format!("{stem}.json"));
        config::load_file(&path).unwrap()
    }

    fn l4_target() -> TargetProfile {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("target_profiles")
            .join("l4_sm89.json");
        load_target(&path).unwrap()
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
        let inferred = infer(&program).unwrap();
        let cfg = build_cfg(&program, &params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        let lib = starter_library();
        let target = l4_target();

        let sfufs = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1]).unwrap();
        let sfuf = &sfufs.per_num_tokens[&1];
        let loops = schedule_workloads(&fuf, &sfufs, &lib);
        let loop_ir = &loops.per_num_tokens[&1];

        let mut scratch = params.bounds.clone();
        scratch.insert("num_tokens".into(), 1);

        let naive = sfuf.predicted_us;
        let aware = loop_cost_us(&fuf, sfuf, loop_ir, &lib, &target, &scratch);

        // aware = sum(per_impl_cost_i) + launches * launch_overhead_us
        // naive = sum(per_impl_cost_i)
        // Both sums use the same per-impl costs (DC discount baked
        // in). The only difference is the per-launch overhead added
        // by loop_cost_us.
        let mut launch_count = 0usize;
        for wave in &loop_ir.waves {
            if wave.is_megakernel {
                launch_count += 1;
            } else {
                launch_count += wave
                    .subgraphs
                    .iter()
                    .filter(|(_, imp_id)| {
                        !matches!(
                            lib.get(*imp_id).launch_kind(),
                            LaunchKind::DeviceCallable | LaunchKind::CooperativeLaunch
                        )
                    })
                    .count();
            }
        }
        let expected_overhead = launch_count as f64 * target.launch_overhead_us;
        assert!(
            (naive - (aware - expected_overhead)).abs() < 1e-6,
            "naive={naive} aware={aware} overhead={expected_overhead} launches={launch_count}"
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
        let inferred = infer(&program).unwrap();
        let cfg = build_cfg(&program, &params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        let lib = starter_library();
        let target = l4_target();

        let sfufs = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1, 512]).unwrap();
        let loops = schedule_workloads(&fuf, &sfufs, &lib);

        let mut scratch = params.bounds.clone();
        for &m in &[1u64, 512] {
            scratch.insert("num_tokens".into(), m);
            let sfuf = &sfufs.per_num_tokens[&m];
            let loop_ir = &loops.per_num_tokens[&m];
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
        let inferred = infer(&program).unwrap();
        let cfg = build_cfg(&program, &params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        let lib = starter_library();
        let target = l4_target();

        let mut sfufs = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1, 512]).unwrap();
        let _before: BTreeMap<u64, f64> = sfufs
            .per_num_tokens
            .iter()
            .map(|(m, sf)| (*m, sf.predicted_us))
            .collect();
        let loops = schedule_workloads(&fuf, &sfufs, &lib);
        refresh_predicted_us(&fuf, &mut sfufs, &loops, &lib, &target, &params.bounds);
        for (_m, sf) in sfufs.per_num_tokens.iter() {
            assert!(sf.predicted_us.is_finite() && sf.predicted_us > 0.0);
            // The refreshed value includes per-launch overhead for
            // host subgraphs and megakernel waves. With DC merging,
            // a megakernel wave pays one launch instead of N, so the
            // refreshed cost can differ from the DP's naive sum in
            // either direction. Just check it's positive and finite.
        }
    }
}
