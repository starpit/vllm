// SPDX-License-Identifier: Apache-2.0
//! Scheduler: turn the SFUF back into a LOOP of waves.
//!
//! A wave groups subgraphs into a single launch. Two criteria
//! allow subgraphs to share a wave:
//!
//! 1. **Independence** — no data dep between them. They run
//!    concurrently (standard BSP superstep).
//! 2. **DeviceCallable chain** — subgraph B depends on A, but
//!    both are `DeviceCallable`. They share a wave because the
//!    megakernel provides ordering via `cg::this_grid().sync()`
//!    between phases — no separate kernel launch needed.
//!
//! The wave assignment algorithm: for each subgraph in topo order,
//! its wave index is `max(wave of predecessors) + bump`, where
//! `bump` is 0 when both the subgraph and the predecessor are DC
//! (they merge into the same wave), and 1 otherwise (new wave).
//!
//! What this pass does NOT do:
//! - pick kernels (that's the solver's job — already done)
//! - emit code (codegen's job)

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::fuf::{Fuf, FufInput};
use crate::impl_lib::{ImplId, ImplementationLibrary, LaunchKind};
use crate::solver::{Assignment, SubgraphId, WorkloadAssignments};

/// A single wave — a group of subgraphs sharing one launch.
#[derive(Clone, Debug)]
pub struct Wave {
    /// The subgraphs in this wave, each paired with the Impl that
    /// realizes it. In a megakernel wave, order is execution order
    /// (dependency-sorted); in a parallel wave, order is arbitrary.
    pub subgraphs: Vec<(SubgraphId, ImplId)>,
    /// True when this wave contains dependent DC subgraphs merged
    /// into a single cooperative kernel launch. Codegen emits one
    /// `__global__` megakernel with grid sync between phases.
    /// False for standard parallel waves.
    pub is_megakernel: bool,
}

/// The LOOP: ordered sequence of waves.
#[derive(Clone, Debug, Default)]
pub struct Loop {
    pub waves: Vec<Wave>,
}

impl Loop {
    pub fn num_waves(&self) -> usize {
        self.waves.len()
    }

    pub fn num_subgraphs(&self) -> usize {
        self.waves.iter().map(|w| w.subgraphs.len()).sum()
    }
}

/// One LOOP per workload point, keyed by `num_tokens`.
#[derive(Clone, Debug, Default)]
pub struct WorkloadLoops {
    pub per_num_tokens: BTreeMap<u64, Loop>,
}

/// Schedule a single SFUF into a LOOP.
///
/// The `lib` parameter is needed to check each impl's `launch_kind`:
/// DeviceCallable subgraphs with DC-only predecessors merge into the
/// same wave (the megakernel provides ordering via grid sync).
pub fn schedule(fuf: &Fuf, sfuf: &Assignment, lib: &ImplementationLibrary) -> Loop {
    if sfuf.num_subgraphs() == 0 {
        return Loop::default();
    }

    // Deps between subgraphs: SG_B depends on SG_A iff some tile
    // in SG_B has a FufInput::Tile pointing at some tile in SG_A
    // (and SG_A != SG_B).
    let mut deps: HashMap<SubgraphId, HashSet<SubgraphId>> = HashMap::new();
    for sg in sfuf.subgraphs() {
        deps.insert(sg, HashSet::new());
    }
    for (tile, sg_consumer) in &sfuf.cover {
        let node = fuf.get(*tile);
        for input in &node.inputs {
            if let FufInput::Tile {
                id: producer_tile, ..
            } = input
                && let Some(sg_producer) = sfuf.subgraph_of(*producer_tile)
                && sg_producer != *sg_consumer
            {
                deps.get_mut(sg_consumer)
                    .expect("entry seeded above")
                    .insert(sg_producer);
            }
        }
    }

    // Pre-compute which subgraphs are DeviceCallable.
    let is_dc: HashMap<SubgraphId, bool> = sfuf
        .subgraphs()
        .map(|sg| {
            let imp = sfuf.impl_of(sg).expect("every subgraph has an impl");
            (sg, lib.get(imp).launch_kind() == LaunchKind::DeviceCallable)
        })
        .collect();

    // Topological wave assignment with DC merging.
    //
    // For each subgraph in topo order, compute its wave index:
    //   wave = max over predecessors of (wave[pred] + bump)
    // where bump = 0 if BOTH this subgraph and the predecessor
    // are DC (they share a wave), bump = 1 otherwise.
    let mut wave_of: HashMap<SubgraphId, u32> = HashMap::new();
    let mut max_wave: u32 = 0;

    let mut ordered: Vec<SubgraphId> = sfuf.subgraphs().collect();
    ordered.sort();

    for sg in &ordered {
        let self_dc = is_dc[sg];
        let depth = deps[sg]
            .iter()
            .map(|d| {
                let pred_wave = wave_of.get(d).copied().unwrap_or(0);
                let pred_dc = is_dc[d];
                if self_dc && pred_dc {
                    // DC→DC: merge into same wave (grid sync handles ordering)
                    pred_wave
                } else {
                    // Any host boundary: new wave
                    pred_wave + 1
                }
            })
            .max()
            .unwrap_or(0);
        wave_of.insert(*sg, depth);
        if depth > max_wave {
            max_wave = depth;
        }
    }

    let n_waves = (max_wave + 1) as usize;
    let mut bins: Vec<Vec<(SubgraphId, ImplId)>> = vec![Vec::new(); n_waves];
    for sg in &ordered {
        let imp = sfuf.impl_of(*sg).expect("every subgraph has an impl");
        let w = wave_of[sg] as usize;
        bins[w].push((*sg, imp));
    }

    Loop {
        waves: bins
            .into_iter()
            .map(|subgraphs| {
                // A wave is a megakernel wave if ALL subgraphs are
                // DeviceCallable (even single-subgraph waves — they get
                // wrapped in a single-phase kernel launch).
                let is_mega = !subgraphs.is_empty() && subgraphs.iter().all(|(sg, _)| is_dc[sg]);
                Wave {
                    subgraphs,
                    is_megakernel: is_mega,
                }
            })
            .collect(),
    }
}

/// Schedule every SFUF in a workload sweep, preserving the `num_tokens` keying.
pub fn schedule_workloads(
    fuf: &Fuf,
    workloads: &WorkloadAssignments,
    lib: &ImplementationLibrary,
) -> WorkloadLoops {
    let per_num_tokens = workloads
        .per_num_tokens
        .iter()
        .map(|(m, sfuf)| (*m, schedule(fuf, sfuf, lib)))
        .collect();
    WorkloadLoops { per_num_tokens }
}

/// Invariant checker: within a non-megakernel wave, no two
/// subgraphs may have a dep edge. Megakernel waves are allowed
/// to contain dependent subgraphs (grid sync handles ordering).
/// Returns (consumer_sg, producer_sg) on violation.
pub fn find_intra_wave_dep_violation(
    loop_ir: &Loop,
    fuf: &Fuf,
    sfuf: &Assignment,
) -> Option<(SubgraphId, SubgraphId)> {
    for wave in &loop_ir.waves {
        if wave.is_megakernel {
            continue; // deps within megakernel waves are intentional
        }
        let wave_set: HashSet<SubgraphId> = wave.subgraphs.iter().map(|(s, _)| *s).collect();
        for (sg, _) in &wave.subgraphs {
            for t in sfuf.tiles_in_subgraph(*sg) {
                let node = fuf.get(t);
                for input in &node.inputs {
                    if let FufInput::Tile { id: producer, .. } = input
                        && let Some(sg_producer) = sfuf.subgraph_of(*producer)
                        && sg_producer != *sg
                        && wave_set.contains(&sg_producer)
                    {
                        return Some((*sg, sg_producer));
                    }
                }
            }
        }
    }
    None
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
    use crate::shape::infer;
    use crate::solver::solve;
    use crate::target::TargetProfile;
    use crate::target::load_file as load_target;
    use std::path::PathBuf;

    fn llama_3_2_1b_params() -> ModelParams {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("model_architectures")
            .join("llama")
            .join("llama-3.2-1b.json");
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

    fn h100_target() -> TargetProfile {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("target_profiles")
            .join("h100_sm90.json");
        load_target(&path).unwrap()
    }

    fn solved_body(src: &str, params: &ModelParams) -> (Fuf, Assignment, ImplementationLibrary) {
        solved_body_target(src, params, &l4_target())
    }

    fn solved_body_h100(
        src: &str,
        params: &ModelParams,
    ) -> (Fuf, Assignment, ImplementationLibrary) {
        solved_body_target(src, params, &h100_target())
    }

    fn solved_body_target(
        src: &str,
        params: &ModelParams,
        target: &TargetProfile,
    ) -> (Fuf, Assignment, ImplementationLibrary) {
        let file: syn::File = syn::parse_str(&format!("fn _c() {{ {src} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).unwrap();
        let program = classify(&ast).unwrap();
        let inferred = infer(&program).unwrap();
        let cfg = build_cfg(&program, params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        let lib = starter_library();
        let workloads = solve(&fuf, &lib, target, &inferred, &params.bounds, &[1]).unwrap();
        let sfuf = workloads.per_num_tokens[&1].clone();
        (fuf, sfuf, lib)
    }

    // Pre-rope attention body: Q/K/V projections without the
    // `rope_append` or `attention` tiles. Keeping rope_append in the
    // body would let `FusedQkvRopeCacheImpl` claim the three Gemms as
    // one subgraph, erasing the Q/K/V parallelism the wave-merging
    // tests below are inspecting. Dropping rope + attention lets the
    // three Gemms stay singleton and share a wave.
    const ATTN_BODY: &str = r#"
        hidden_states = embed(input_ids, embed_tokens);
        for layer in 0..num_hidden_layers {
            normed = rmsnorm(hidden_states, input_layernorm[layer]);
            q = gemm(normed, self_attn.q_proj[layer]);
            k = gemm(normed, self_attn.k_proj[layer]);
            v = gemm(normed, self_attn.v_proj[layer]);
            oproj = gemm(q, self_attn.o_proj[layer]);
            hidden_states = add(oproj, hidden_states);
        }
        // Post-loop final norm so the last iteration's Add has a
        // downstream RmsNorm consumer — otherwise the solver has no
        // coverage for that Add (no standalone Add impl exists).
        final_norm = rmsnorm(hidden_states, norm);
    "#;

    #[test]
    fn loop_is_shorter_than_one_wave_per_subgraph() {
        let (fuf, sfuf, lib) = solved_body(ATTN_BODY, &llama_3_2_1b_params());
        let loop_ir = schedule(&fuf, &sfuf, &lib);

        // Every subgraph in the SFUF appears exactly once in the LOOP.
        assert_eq!(loop_ir.num_subgraphs(), sfuf.num_subgraphs());

        // Fewer waves than subgraphs: q/k/v gemms merge (they all
        // read the same `normed` and are mutually independent).
        assert!(
            loop_ir.num_waves() < sfuf.num_subgraphs(),
            "waves={} subgraphs={} — expected wave merging",
            loop_ir.num_waves(),
            sfuf.num_subgraphs(),
        );
    }

    #[test]
    fn no_intra_wave_deps() {
        let (fuf, sfuf, lib) = solved_body(ATTN_BODY, &llama_3_2_1b_params());
        let loop_ir = schedule(&fuf, &sfuf, &lib);
        assert!(
            find_intra_wave_dep_violation(&loop_ir, &fuf, &sfuf).is_none(),
            "schedule invariant broken",
        );
    }

    #[test]
    fn parallel_qkv_gemms_share_a_wave() {
        let (fuf, sfuf, lib) = solved_body(
            r#"
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..1 {
                normed = rmsnorm(hidden_states, input_layernorm[layer]);
                q = gemm(normed, self_attn.q_proj[layer]);
                k = gemm(normed, self_attn.k_proj[layer]);
                v = gemm(normed, self_attn.v_proj[layer]);
                oproj = gemm(q, self_attn.o_proj[layer]);
                hidden_states = add(oproj, hidden_states);
            }
            final_norm = rmsnorm(hidden_states, norm);
            "#,
            &llama_3_2_1b_params(),
        );
        let loop_ir = schedule(&fuf, &sfuf, &lib);

        // The q/k/v gemms read `normed` and are mutually independent,
        // so they can share a wave. With DC GEMM variants, some or all
        // may be DC and merge into a megakernel wave with other DC ops.
        // Verify: at least one wave contains ≥2 gemm subgraphs (the
        // independent q/k/v can be scheduled together).
        use crate::classified::OpKind;
        let max_gemms_in_wave = loop_ir
            .waves
            .iter()
            .map(|wave| {
                wave.subgraphs
                    .iter()
                    .filter(|(sg, _)| {
                        sfuf.tiles_in_subgraph(*sg)
                            .iter()
                            .any(|t| fuf.get(*t).op == OpKind::Gemm)
                    })
                    .count()
            })
            .max()
            .unwrap_or(0);
        assert!(
            max_gemms_in_wave >= 2,
            "expected at least 2 independent gemms in one wave; max was {max_gemms_in_wave}"
        );
    }

    #[test]
    fn workload_sweep_produces_one_loop_per_point() {
        let params = llama_3_2_1b_params();
        let file: syn::File = syn::parse_str(
            r#"fn _c() {
                hidden_states = embed(input_ids, embed_tokens);
                for layer in 0..num_hidden_layers {
                    normed = rmsnorm(hidden_states, input_layernorm[layer]);
                    q = gemm(normed, self_attn.q_proj[layer]);
                    oproj = gemm(q, self_attn.o_proj[layer]);
                    hidden_states = add(oproj, hidden_states);
                }
                final_norm = rmsnorm(hidden_states, norm);
            }"#,
        )
        .unwrap();
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

        let points = [1u64, 64, 4096];
        let workloads = solve(&fuf, &lib, &target, &inferred, &params.bounds, &points).unwrap();
        let loops = schedule_workloads(&fuf, &workloads, &lib);

        assert_eq!(loops.per_num_tokens.len(), points.len());
        for &m in &points {
            let loop_ir = loops
                .per_num_tokens
                .get(&m)
                .unwrap_or_else(|| panic!("no loop at m={m}"));
            let sfuf = &workloads.per_num_tokens[&m];
            assert_eq!(loop_ir.num_subgraphs(), sfuf.num_subgraphs());
            assert!(find_intra_wave_dep_violation(loop_ir, &fuf, sfuf).is_none());
        }
    }

    /// Llama on L4 has no consecutive DC subgraphs (every DC
    /// elementwise is separated by a host GEMM). Verify zero
    /// megakernel waves.
    #[test]
    fn llama_l4_produces_megakernel_waves() {
        let params = llama_3_2_1b_params();
        let (fuf, sfuf, lib) = solved_body(
            r#"
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
            "#,
            &params,
        );
        let loop_ir = schedule(&fuf, &sfuf, &lib);
        assert_eq!(loop_ir.num_subgraphs(), sfuf.num_subgraphs());

        let mega_count = loop_ir.waves.iter().filter(|w| w.is_megakernel).count();
        // With DC wrappers for CUTLASS GEMMs, all ops can be DC,
        // so megakernel waves form across the entire loop body.
        assert!(
            mega_count > 0,
            "expected megakernel waves on L4 Llama with DC GEMM; got {mega_count}"
        );
    }

    /// A body with consecutive rmsnorm ops produces megakernel waves
    /// because dc_rmsnorm chains merge into a single wave.
    #[test]
    fn consecutive_dc_ops_produce_megakernel_waves() {
        let params = llama_3_2_1b_params();
        // Two consecutive rmsnorms with no GEMM between them.
        // The solver picks dc_rmsnorm_ref for both, and the
        // scheduler merges them into a megakernel wave.
        let (fuf, sfuf, lib) = solved_body(
            r#"
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..num_hidden_layers {
                normed1 = rmsnorm(hidden_states, input_layernorm[layer]);
                hidden_states = rmsnorm(normed1, post_attention_layernorm[layer]);
            }
            normed = rmsnorm(hidden_states, norm);
            logits = gemm(normed, lm_head);
            "#,
            &params,
        );
        let loop_ir = schedule(&fuf, &sfuf, &lib);
        assert_eq!(loop_ir.num_subgraphs(), sfuf.num_subgraphs());

        let mega_count = loop_ir.waves.iter().filter(|w| w.is_megakernel).count();
        let mega_sgs: usize = loop_ir
            .waves
            .iter()
            .filter(|w| w.is_megakernel)
            .map(|w| w.subgraphs.len())
            .sum();
        assert!(
            mega_count > 0,
            "expected megakernel waves for consecutive rmsnorms; got 0 \
             out of {} total waves",
            loop_ir.num_waves(),
        );
        eprintln!(
            "consecutive DC: {mega_count} mega waves ({mega_sgs} DC subgraphs), \
             {} total waves, {} total subgraphs",
            loop_ir.num_waves(),
            sfuf.num_subgraphs(),
        );
    }

    /// On H100 (sm90), TK GEMM/GEMV are natively DeviceCallable and
    /// cost less than cuBLAS (cuBLAS_cost - launch_overhead). Combined
    /// with DC wrappers for elementwise ops and TK attention, this
    /// means EVERY op in the LLaMA decoder layer is DeviceCallable.
    /// The scheduler should produce 100% megakernel waves.
    #[test]
    fn h100_llama_100_percent_megakernel() {
        let params = llama_3_2_1b_params();
        let (fuf, sfuf, lib) = solved_body_h100(
            r#"
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
            "#,
            &params,
        );
        let loop_ir = schedule(&fuf, &sfuf, &lib);

        // Count non-megakernel subgraphs. embed_ref is the only
        // acceptable standalone — it's a gather lookup that runs once
        // before the loop body. All GEMM/GEMV/elementwise/attention
        // ops must be DeviceCallable.
        let non_mega_sgs: Vec<_> = loop_ir
            .waves
            .iter()
            .filter(|w| !w.is_megakernel)
            .flat_map(|w| &w.subgraphs)
            .collect();
        let non_embed: Vec<_> = non_mega_sgs
            .iter()
            .filter(|sg| lib.get(sg.1).name() != "embed_ref")
            .collect();

        assert!(
            non_embed.is_empty(),
            "H100 LLaMA: all non-embed ops should be DeviceCallable, \
             but {} standalone subgraphs remain: {:?}",
            non_embed.len(),
            non_embed
                .iter()
                .map(|sg| format!("impl={}", lib.get(sg.1).name()))
                .collect::<Vec<_>>(),
        );

        let mega_count = loop_ir.waves.iter().filter(|w| w.is_megakernel).count();
        let total = sfuf.num_subgraphs();
        let mega_sgs: usize = loop_ir
            .waves
            .iter()
            .filter(|w| w.is_megakernel)
            .map(|w| w.subgraphs.len())
            .sum();
        eprintln!(
            "H100 megakernel: {mega_sgs}/{total} subgraphs in {mega_count} mega waves \
             (embed is standalone)"
        );
    }

    #[test]
    fn empty_sfuf_empty_loop() {
        let fuf = Fuf { nodes: Vec::new() };
        let sfuf = Assignment::default();
        let lib = starter_library();
        let loop_ir = schedule(&fuf, &sfuf, &lib);
        assert_eq!(loop_ir.num_waves(), 0);
        assert_eq!(loop_ir.num_subgraphs(), 0);
    }
}
