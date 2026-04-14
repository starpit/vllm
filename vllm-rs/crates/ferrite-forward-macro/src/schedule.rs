// SPDX-License-Identifier: Apache-2.0
//! Phase 8: BSP schedule.
//!
//! Given the solver's per-tile impl picks, produce a sequence of
//! steps. Each step holds tiles that run concurrently; steps run
//! sequentially. Within a step, no two tiles may have a dep edge
//! between them; across steps, tile dependence dictates ordering.
//!
//! The MVP computes topological layers over the FUF: each tile's
//! layer is `1 + max(input layer)`. Tiles in the same layer are
//! mutually independent (by construction) and go in the same step.
//! This *is* the step-merge optimization relative to a naive
//! one-tile-per-step schedule: merging adjacent independent tiles
//! eliminates launch overhead without violating deps.
//!
//! Further merging — packing across layers when claim-masks and
//! resource budgets allow — stays for later phases, when the
//! library has multiple impls per op with non-trivial claim_masks
//! and the potential wins are real.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::ImplId;
use crate::solver::Assignment;

/// A BSP step: one or more tiles that run concurrently.
#[derive(Clone, Debug)]
pub struct Step {
    pub tiles: Vec<(TileId, ImplId)>,
}

/// An ordered sequence of steps.
#[derive(Clone, Debug)]
pub struct Schedule {
    pub steps: Vec<Step>,
}

impl Schedule {
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    pub fn num_tiles(&self) -> usize {
        self.steps.iter().map(|s| s.tiles.len()).sum()
    }
}

/// Compute a BSP schedule from a FUF + assignment.
///
/// Tiles are grouped by topological layer. Within a layer, tiles
/// execute concurrently; across layers they execute sequentially.
/// Order within a step follows FUF tile order (deterministic).
pub fn schedule(fuf: &Fuf, assignment: &Assignment) -> Schedule {
    if fuf.is_empty() {
        return Schedule { steps: Vec::new() };
    }

    // Topological layer per tile: 1 + max(layer of any Tile-input
    // producer).
    let mut layer: HashMap<TileId, u32> = HashMap::with_capacity(fuf.len());
    let mut max_layer: u32 = 0;
    for node in &fuf.nodes {
        let mut depth = 0;
        for input in &node.inputs {
            if let FufInput::Tile { id, .. } = input {
                let d = layer.get(id).copied().unwrap_or(0) + 1;
                if d > depth {
                    depth = d;
                }
            }
        }
        layer.insert(node.id, depth);
        if depth > max_layer {
            max_layer = depth;
        }
    }

    let n_layers = (max_layer + 1) as usize;
    let mut bins: Vec<Vec<(TileId, ImplId)>> = vec![Vec::new(); n_layers];
    for node in &fuf.nodes {
        let l = layer[&node.id] as usize;
        let imp = assignment
            .tile_to_impl
            .get(&node.id)
            .copied()
            .expect("assignment has entry for every tile");
        bins[l].push((node.id, imp));
    }

    Schedule {
        steps: bins.into_iter().map(|tiles| Step { tiles }).collect(),
    }
}

/// Invariant check: within a single step, no tile depends on
/// another. Returns a (violating_tile, dep_target) pair if the
/// invariant is broken, otherwise `None`.
pub fn find_intra_step_dep_violation(schedule: &Schedule, fuf: &Fuf) -> Option<(TileId, TileId)> {
    for step in &schedule.steps {
        let step_tiles: std::collections::HashSet<TileId> =
            step.tiles.iter().map(|(t, _)| *t).collect();
        for (tile, _) in &step.tiles {
            let node = fuf.get(*tile);
            for input in &node.inputs {
                if let FufInput::Tile { id, .. } = input
                    && step_tiles.contains(id)
                {
                    return Some((*tile, *id));
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

    fn plan_body(src: &str, params: &ModelParams) -> (Fuf, Assignment) {
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
        let mut bounds = params.bounds.clone();
        bounds.insert("num_tokens".into(), 1);
        let assignment = solve(&fuf, &lib, &target, &bounds).unwrap();
        (fuf, assignment)
    }

    #[test]
    fn schedule_is_shorter_than_one_per_tile() {
        let (fuf, assignment) = plan_body(
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
            }
            "#,
            &llama_3_2_1b_params(),
        );

        let sched = schedule(&fuf, &assignment);

        // Every tile in the FUF appears exactly once in the schedule.
        assert_eq!(sched.num_tiles(), fuf.len());

        // Strictly fewer steps than tiles: q/k/v gemms merge into
        // one step per iteration (they're mutually independent
        // reads of `normed`).
        assert!(
            sched.num_steps() < fuf.len(),
            "schedule has {} steps for {} tiles — expected merging",
            sched.num_steps(),
            fuf.len(),
        );
    }

    #[test]
    fn no_intra_step_deps() {
        let (fuf, assignment) = plan_body(
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
            }
            "#,
            &llama_3_2_1b_params(),
        );
        let sched = schedule(&fuf, &assignment);
        assert!(
            find_intra_step_dep_violation(&sched, &fuf).is_none(),
            "schedule invariant broken: two tiles in same step have a dep edge",
        );
    }

    #[test]
    fn parallel_qkv_gemms_share_step() {
        // q/k/v gemms all read `normed` and are mutually independent.
        // The schedule should put them in a single step.
        let (fuf, assignment) = plan_body(
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
            "#,
            &llama_3_2_1b_params(),
        );
        let sched = schedule(&fuf, &assignment);

        // Find the step that contains all three q/k/v gemms.
        // (They all read `normed` from the preceding rmsnorm, so
        // they share a topological layer.)
        let mut found_qkv_step = false;
        for step in &sched.steps {
            let gemm_count = step
                .tiles
                .iter()
                .filter(|(t, _)| matches!(fuf.get(*t).op, crate::classified::OpKind::Gemm))
                .count();
            if gemm_count == 3 {
                found_qkv_step = true;
                break;
            }
        }
        assert!(found_qkv_step, "expected q/k/v gemms to share one step");
    }

    #[test]
    fn empty_fuf_empty_schedule() {
        let fuf = Fuf { nodes: Vec::new() };
        let assignment = Assignment {
            tile_to_impl: HashMap::new(),
            predicted_us: 0.0,
        };
        let sched = schedule(&fuf, &assignment);
        assert_eq!(sched.num_steps(), 0);
        assert_eq!(sched.num_tiles(), 0);
    }
}
