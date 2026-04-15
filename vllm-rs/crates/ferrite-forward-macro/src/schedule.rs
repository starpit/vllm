// SPDX-License-Identifier: Apache-2.0
//! Scheduler: turn the SFUF back into a LOOP of waves.
//!
//! A wave (== BSP superstep) is a set of subgraphs with no
//! dependence on each other; subgraphs within a wave are
//! concurrent, waves execute in sequence. Topological layering
//! over subgraphs: each subgraph's wave index is `1 + max(wave of
//! any subgraph it depends on)`.
//!
//! A dep between subgraphs A and B exists when some tile in B has
//! a `FufInput::Tile` edge to a tile in A. Within the same
//! subgraph, internal deps are irrelevant to scheduling (the Impl
//! handles them internally).
//!
//! What this pass does NOT do:
//! - pick kernels (that's the solver's job — already done)
//! - decide launch mode (that's a tag on each Impl, read by
//!   codegen, NOT a choice the scheduler makes)
//! - emit code (codegen's job)

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::fuf::{Fuf, FufInput};
use crate::impl_lib::ImplId;
use crate::solver::{Assignment, SubgraphId, WorkloadAssignments};

/// A single BSP wave — mutually-independent subgraphs.
#[derive(Clone, Debug)]
pub struct Wave {
    /// The subgraphs in this wave, each paired with the Impl that
    /// realizes it. Order within a wave is irrelevant for
    /// scheduling (subgraphs are concurrent); codegen may pick an
    /// emission order.
    pub subgraphs: Vec<(SubgraphId, ImplId)>,
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
pub fn schedule(fuf: &Fuf, sfuf: &Assignment) -> Loop {
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

    // Topological wave assignment. Visit subgraphs in id order
    // (solver allocates ids in topo order of first-claimed tile,
    // so deps flow forward — but compute wave via max of predecessor
    // waves to be safe for multi-tile claims).
    let mut wave_of: HashMap<SubgraphId, u32> = HashMap::new();
    let mut max_wave: u32 = 0;

    // Sort subgraphs by id for deterministic iteration. Topological
    // order by id holds because the solver's forward pass creates
    // each subgraph only after all its prior tiles were committed.
    let mut ordered: Vec<SubgraphId> = sfuf.subgraphs().collect();
    ordered.sort();

    for sg in &ordered {
        let depth = deps[sg]
            .iter()
            .map(|d| wave_of.get(d).copied().unwrap_or(0) + 1)
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
            .map(|subgraphs| Wave { subgraphs })
            .collect(),
    }
}

/// Schedule every SFUF in a workload sweep, preserving the `num_tokens` keying.
pub fn schedule_workloads(fuf: &Fuf, workloads: &WorkloadAssignments) -> WorkloadLoops {
    let per_num_tokens = workloads
        .per_num_tokens
        .iter()
        .map(|(m, sfuf)| (*m, schedule(fuf, sfuf)))
        .collect();
    WorkloadLoops { per_num_tokens }
}

/// Invariant checker: within a single wave, no two subgraphs have
/// a dep edge. Returns (consumer_sg, producer_sg) on violation.
pub fn find_intra_wave_dep_violation(
    loop_ir: &Loop,
    fuf: &Fuf,
    sfuf: &Assignment,
) -> Option<(SubgraphId, SubgraphId)> {
    for wave in &loop_ir.waves {
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

    fn solved_body(src: &str, params: &ModelParams) -> (Fuf, Assignment) {
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
        let target = l4_target();
        let workloads = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1]).unwrap();
        let sfuf = workloads.per_num_tokens[&1].clone();
        (fuf, sfuf)
    }

    const ATTN_BODY: &str = r#"
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
        }
        // Post-loop final norm so the last iteration's Add has a
        // downstream RmsNorm consumer — otherwise the solver has no
        // coverage for that Add (no standalone Add impl exists).
        final_norm = rmsnorm(hidden_states, norm);
    "#;

    #[test]
    fn loop_is_shorter_than_one_wave_per_subgraph() {
        let (fuf, sfuf) = solved_body(ATTN_BODY, &llama_3_2_1b_params());
        let loop_ir = schedule(&fuf, &sfuf);

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
        let (fuf, sfuf) = solved_body(ATTN_BODY, &llama_3_2_1b_params());
        let loop_ir = schedule(&fuf, &sfuf);
        assert!(
            find_intra_wave_dep_violation(&loop_ir, &fuf, &sfuf).is_none(),
            "schedule invariant broken",
        );
    }

    #[test]
    fn parallel_qkv_gemms_share_a_wave() {
        let (fuf, sfuf) = solved_body(
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
        let loop_ir = schedule(&fuf, &sfuf);

        // Find the wave containing all three q/k/v gemms (they
        // read `normed` and are mutually independent). Starter
        // library is 1 tile = 1 subgraph, so look at each
        // subgraph's single claimed tile's op.
        use crate::classified::OpKind;
        let mut found = false;
        for wave in &loop_ir.waves {
            let gemm_count = wave
                .subgraphs
                .iter()
                .filter(|(sg, _)| {
                    sfuf.tiles_in_subgraph(*sg)
                        .iter()
                        .any(|t| fuf.get(*t).op == OpKind::Gemm)
                })
                .count();
            if gemm_count == 3 {
                found = true;
                break;
            }
        }
        assert!(found, "expected q/k/v gemms to share one wave");
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
        let loops = schedule_workloads(&fuf, &workloads);

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

    #[test]
    fn empty_sfuf_empty_loop() {
        let fuf = Fuf { nodes: Vec::new() };
        let sfuf = Assignment::default();
        let loop_ir = schedule(&fuf, &sfuf);
        assert_eq!(loop_ir.num_waves(), 0);
        assert_eq!(loop_ir.num_subgraphs(), 0);
    }
}
