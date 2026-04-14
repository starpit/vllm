// SPDX-License-Identifier: Apache-2.0
//! Phase 7: solver — pick one [`Implementation`] per tile.
//!
//! For Phase 7 MVP this is a per-tile cost-greedy picker: for each
//! tile, select the cheapest candidate Implementation whose OpKind
//! matches. No DP, no backtracking, no cross-tile interaction —
//! honest about what it is.
//!
//! When the library grows multiple impls per op with non-trivial
//! claim_mask interactions, this becomes insufficient and we
//! upgrade to DP over (position, claim_state). Until then, running
//! DP would be theatre: with one candidate per op, every solver
//! is equivalent.
//!
//! Phase 8 does the actual combinatorial work via step-merging.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::collections::HashMap;

use crate::fuf::{Fuf, FufInput, FufNode, TileId};
use crate::impl_lib::{CostCtx, ImplId, ImplementationLibrary};
use crate::shape::Shape;
use crate::target::TargetProfile;

#[derive(Clone, Debug)]
pub struct Assignment {
    pub tile_to_impl: HashMap<TileId, ImplId>,
    /// Sum of per-tile estimated costs in microseconds. For tiles
    /// with unresolved shape dims the contribution is 0 — cost
    /// underestimates, but it's still useful for relative impl
    /// picks. Phase 8 refines.
    pub predicted_us: f64,
}

#[derive(Debug)]
pub enum SolveError {
    NoCandidate {
        tile: TileId,
        op: crate::classified::OpKind,
    },
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCandidate { tile, op } => write!(
                f,
                "no implementation available for tile {} op {:?}",
                tile.0, op
            ),
        }
    }
}

impl std::error::Error for SolveError {}

pub fn solve(
    fuf: &Fuf,
    lib: &ImplementationLibrary,
    target: &TargetProfile,
    bounds: &BTreeMap<String, u64>,
) -> Result<Assignment, SolveError> {
    let mut tile_to_impl = HashMap::with_capacity(fuf.len());
    let mut total = 0.0_f64;

    for node in &fuf.nodes {
        let candidates = lib.candidates(node.op);
        if candidates.is_empty() {
            return Err(SolveError::NoCandidate {
                tile: node.id,
                op: node.op,
            });
        }

        // Input shapes: look up the producing tile's output for
        // each Tile input; for Weight/Extern inputs shape info is
        // not tracked per-input here (they're either runtime-dim
        // externs or shape-known weights that were unified earlier
        // in shape inference). For MVP, assemble input shapes from
        // whatever is reachable.
        let input_shapes = resolve_input_shapes(fuf, node);

        // Pick the cheapest candidate.
        let mut best: Option<(ImplId, f64)> = None;
        for &cand in candidates {
            let ctx = CostCtx {
                input_shapes: &input_shapes,
                output_shapes: &node.outputs,
                target,
                bounds,
            };
            let cost = (lib.get(cand).cost_fn)(&ctx).unwrap_or(f64::INFINITY);
            if best.map(|(_, b)| cost < b).unwrap_or(true) {
                best = Some((cand, cost));
            }
        }
        let (chosen, cost) = best.expect("non-empty candidates");
        tile_to_impl.insert(node.id, chosen);
        if cost.is_finite() {
            total += cost;
        }
    }

    Ok(Assignment {
        tile_to_impl,
        predicted_us: total,
    })
}

fn resolve_input_shapes(fuf: &Fuf, node: &FufNode) -> Vec<Shape> {
    node.inputs
        .iter()
        .map(|inp| match inp {
            FufInput::Tile { id, slot } => {
                let upstream = fuf.get(*id);
                upstream
                    .outputs
                    .get(*slot as usize)
                    .cloned()
                    .unwrap_or_default()
            }
            // Weights and externs don't carry a shape at tile-input
            // resolution time. Ops that need them use the op
            // signature's constraints (which already ran in Phase 4).
            FufInput::Weight { .. } | FufInput::Extern { .. } => Vec::new(),
        })
        .collect()
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

    fn build_fuf(src: &str, params: &ModelParams) -> Fuf {
        let file: syn::File = syn::parse_str(&format!("fn _c() {{ {src} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).unwrap();
        let program = classify(&ast).unwrap();
        let inferred = infer(&program).unwrap();
        let cfg = build_cfg(&program, params).unwrap();
        unroll(&cfg, &inferred).unwrap()
    }

    #[test]
    fn every_tile_gets_an_impl() {
        let fuf = build_fuf(
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
        let lib = starter_library();
        let target = l4_target();
        let mut bounds = llama_3_2_1b_params().bounds.clone();
        bounds.insert("num_tokens".into(), 1);

        let t0 = std::time::Instant::now();
        let assignment = solve(&fuf, &lib, &target, &bounds).expect("solve");
        let elapsed_ms = t0.elapsed().as_millis();

        // Every tile must have an impl.
        assert_eq!(
            assignment.tile_to_impl.len(),
            fuf.len(),
            "every tile assigned"
        );

        // PLAN's deadline: solve < 100 ms on realistic Llama graph.
        assert!(
            elapsed_ms < 100,
            "solve took {elapsed_ms} ms, budget 100 ms",
        );

        // Total cost estimate should be positive and finite.
        assert!(
            assignment.predicted_us > 0.0 && assignment.predicted_us.is_finite(),
            "got {}",
            assignment.predicted_us,
        );
    }

    #[test]
    fn missing_candidate_is_an_error() {
        let fuf = build_fuf(
            "hidden_states = embed(input_ids, embed_tokens);",
            &llama_3_2_1b_params(),
        );
        let lib = ImplementationLibrary::new(); // empty
        let target = l4_target();
        let bounds = llama_3_2_1b_params().bounds.clone();

        let err = solve(&fuf, &lib, &target, &bounds).unwrap_err();
        assert!(matches!(err, SolveError::NoCandidate { .. }));
    }
}
