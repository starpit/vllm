// SPDX-License-Identifier: Apache-2.0
//! Solver: match kernels to FUF tiles, produce the SFUF.
//!
//! Algorithm ported from
//! `ferrite-solver/src/lowering/solver/dp.rs`. Topological forward
//! pass with a `claimed[]` bitmap: at each unclaimed tile in topo
//! order, pick the cheapest [`Implementation`] whose [`MatchInfo`]
//! doesn't conflict with previously-committed claims. Multi-tile
//! matches collapse their claimed tiles into one subgraph.
//!
//! NOT ported from the old DP (llama-specific hacks — see PLAN.md
//! NON-reuse):
//!
//! - The `matches!(node.kind, TileKind::ResidualAdd)` auto-claim
//!   fallback for tiles no impl matched. An unmatched tile is a
//!   library bug, not something to paper over.
//! - The `TileKind::GemmQ/K/V/...` taxonomy and any code that
//!   matched on it. Our OpKind is uniform.
//! - `weight_name: String` and weight-name substring checks.
//!
//! The workload sweep stays: for each `num_tokens` the caller asks
//! about, we produce one SFUF. Codegen coalesces adjacent-equal
//! SFUFs into match arms.
//!
//! Output: a [`WorkloadAssignments`] keyed by `num_tokens`. Each
//! [`Assignment`] (== SFUF) has `cover: tile → SubgraphId`, `impls:
//! SubgraphId → ImplId`, and `predicted_us`.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

use crate::classified::OpKind;
use crate::fuf::{Fuf, FufInput, FufNode, TileId};
use crate::impl_lib::{CostCtx, ImplId, ImplementationLibrary, MatchInfo};
use crate::shape::{Inferred, Shape, extern_shape};
use crate::target::TargetProfile;

/// Stable identifier for one claimed subgraph within an SFUF. Each
/// commit of "these tiles are claimed by this impl" allocates a
/// fresh `SubgraphId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubgraphId(pub u32);

/// An SFUF: the FUF with every tile bound to a subgraph and every
/// subgraph bound to one Impl.
#[derive(Clone, Debug, Default)]
pub struct Assignment {
    /// Tile → which subgraph claims it.
    pub cover: HashMap<TileId, SubgraphId>,
    /// Subgraph → which Impl realizes it.
    pub impls: HashMap<SubgraphId, ImplId>,
    /// Sum of per-subgraph cost estimates in microseconds at this
    /// workload point.
    pub predicted_us: f64,
}

impl Assignment {
    pub fn is_cover_complete(&self, num_tiles: usize) -> bool {
        self.cover.len() == num_tiles
    }

    pub fn subgraph_of(&self, tile: TileId) -> Option<SubgraphId> {
        self.cover.get(&tile).copied()
    }

    pub fn impl_of(&self, sg: SubgraphId) -> Option<ImplId> {
        self.impls.get(&sg).copied()
    }

    pub fn subgraphs(&self) -> impl Iterator<Item = SubgraphId> + '_ {
        self.impls.keys().copied()
    }

    pub fn tiles_in_subgraph(&self, sg: SubgraphId) -> Vec<TileId> {
        let mut v: Vec<TileId> = self
            .cover
            .iter()
            .filter_map(|(t, s)| if *s == sg { Some(*t) } else { None })
            .collect();
        v.sort();
        v
    }

    pub fn num_subgraphs(&self) -> usize {
        self.impls.len()
    }
}

/// Result of a full workload sweep: one SFUF per caller-supplied
/// `num_tokens` value, keyed by that value.
#[derive(Clone, Debug, Default)]
pub struct WorkloadAssignments {
    pub per_num_tokens: BTreeMap<u64, Assignment>,
}

#[derive(Debug)]
pub enum SolveError {
    /// No Impl in the library matched the given tile at this
    /// workload point. Library is incomplete, or a workload/target
    /// filter excluded the only viable Impl. Never papered over by
    /// an auto-claim fallback.
    UnclaimedTile {
        tile: TileId,
        op: OpKind,
        num_tokens: u64,
    },
    /// An Impl's cost function returned None despite having a
    /// match and passing applicability filters. Means either (a)
    /// shape inference left a Var, or (b) a bug in the cost fn.
    /// Silently skipping the Impl would mask the bug — hard error.
    UnreachableCost {
        tile: TileId,
        op: OpKind,
        impl_id: ImplId,
        num_tokens: u64,
    },
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnclaimedTile {
                tile,
                op,
                num_tokens,
            } => write!(
                f,
                "no Impl in the library matched tile {} op {:?} at num_tokens={num_tokens}",
                tile.0, op
            ),
            Self::UnreachableCost {
                tile,
                op,
                impl_id,
                num_tokens,
            } => write!(
                f,
                "impl {} could not cost tile {} op {:?} at num_tokens={num_tokens} \
                 (cost_fn returned None despite invariant that shapes are closed)",
                impl_id.0, tile.0, op
            ),
        }
    }
}

impl std::error::Error for SolveError {}

/// Solve each `num_tokens` point independently. `bounds` supplies
/// every symbolic bound except `num_tokens`; the solver overwrites
/// it per point.
pub fn solve(
    fuf: &Fuf,
    lib: &ImplementationLibrary,
    target: &TargetProfile,
    inferred: &Inferred,
    bounds: &BTreeMap<String, u64>,
    num_tokens_points: &[u64],
) -> Result<WorkloadAssignments, SolveError> {
    let mut per_num_tokens: BTreeMap<u64, Assignment> = BTreeMap::new();
    let mut scratch = bounds.clone();

    for &m in num_tokens_points {
        scratch.insert("num_tokens".into(), m);
        let assignment = solve_one(fuf, lib, target, inferred, &scratch, m)?;
        per_num_tokens.insert(m, assignment);
    }

    Ok(WorkloadAssignments { per_num_tokens })
}

/// One pass of the DP over the whole FUF at a single workload
/// point. Returns the SFUF.
fn solve_one(
    fuf: &Fuf,
    lib: &ImplementationLibrary,
    target: &TargetProfile,
    inferred: &Inferred,
    bounds: &BTreeMap<String, u64>,
    num_tokens: u64,
) -> Result<Assignment, SolveError> {
    let n = fuf.len();
    if n == 0 {
        return Ok(Assignment {
            cover: HashMap::new(),
            impls: HashMap::new(),
            predicted_us: 0.0,
        });
    }

    // ── Phase 1: precompute candidate matches per tile ──
    //
    // For each tile (in topological order), enumerate every library
    // Impl whose target/workload filters admit this context. Call
    // matches() at this seed tile; if it returns Some, cost the
    // match. Sort cheapest first so the forward pass picks greedily
    // (greedy is optimal over the local-claim structure of the
    // library — see old dp.rs notes).
    let mut matches_at: Vec<Vec<(ImplId, MatchInfo, f64)>> = vec![Vec::new(); n];

    for (i, node) in fuf.nodes.iter().enumerate() {
        let input_shapes = resolve_input_shapes(fuf, node, inferred);
        let ctx = CostCtx {
            input_shapes: &input_shapes,
            output_shapes: &node.outputs,
            target,
            bounds,
        };

        for (imp_id, imp) in lib.iter_enumerated() {
            if !imp.target_filter.matches(target) {
                continue;
            }
            if !imp.workload_constraint.accepts(num_tokens) {
                continue;
            }
            let Some(info) = imp.matches(node.id, fuf) else {
                continue;
            };
            let cost = (imp.cost_fn)(&ctx).ok_or(SolveError::UnreachableCost {
                tile: node.id,
                op: node.op,
                impl_id: imp_id,
                num_tokens,
            })?;
            matches_at[i].push((imp_id, info, cost));
        }

        matches_at[i].sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));
    }

    // ── Phase 2: topological forward pass ──
    //
    // Walk tiles in order. At each unclaimed tile, pick the cheapest
    // matching Impl whose claim doesn't overlap previously-committed
    // claims. Commit: allocate a subgraph id, mark claimed, record.
    //
    // Unmatched tile (after filtering) → hard UnclaimedTile error.
    // No auto-claim by tile kind; no silent skip.
    let mut claimed: Vec<bool> = vec![false; n];
    let mut assignment = Assignment::default();
    let mut next_sg: u32 = 0;
    let mut total = 0.0_f64;

    for i in 0..n {
        if claimed[i] {
            continue;
        }
        let node = &fuf.nodes[i];

        let best = matches_at[i]
            .iter()
            .find(|(_, info, _)| info.claimed_tiles.iter().all(|t| !claimed[t.0 as usize]));

        let Some((imp_id, info, cost)) = best else {
            return Err(SolveError::UnclaimedTile {
                tile: node.id,
                op: node.op,
                num_tokens,
            });
        };

        let sg = SubgraphId(next_sg);
        next_sg += 1;

        for t in &info.claimed_tiles {
            claimed[t.0 as usize] = true;
            assignment.cover.insert(*t, sg);
        }
        assignment.impls.insert(sg, *imp_id);
        total += cost;
    }

    // Invariant: every tile is claimed. Topo forward pass + "every
    // seed has at least one matching Impl" should guarantee this.
    debug_assert!(
        claimed.iter().all(|&c| c),
        "DP left {} tiles unclaimed",
        claimed.iter().filter(|&&c| !c).count(),
    );

    assignment.predicted_us = total;
    Ok(assignment)
}

/// Resolve each tile input's shape for the cost function.
///
/// - `Tile` inputs read the producing tile's output slot.
/// - `Weight` inputs read from [`Inferred::weights`].
/// - `Extern` inputs read from [`extern_shape`].
fn resolve_input_shapes(fuf: &Fuf, node: &FufNode, inferred: &Inferred) -> Vec<Shape> {
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
            FufInput::Weight { id, .. } => inferred.weights.get(id).cloned().unwrap_or_default(),
            FufInput::Extern { kind, .. } => extern_shape(*kind),
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
    use crate::impl_lib::{
        CostCtx, Implementation, ImplementationLibrary, LaunchKind, Layout, MatchInfo,
        TargetFilter, WorkloadConstraint, starter_library,
    };
    use crate::parse::parse_block;
    use crate::shape::infer;
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

    fn build(src: &str, params: &ModelParams) -> (Fuf, Inferred) {
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
        (fuf, inferred)
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
            up = gemm(normed2, mlp.up_proj[layer]);
            down = gemm(up, mlp.down_proj[layer]);
            hidden_states = add(down, hidden_states);
        }
        normed = rmsnorm(hidden_states, norm);
        logits = gemm(normed, lm_head);
    "#;

    const WORKLOAD_POINTS: [u64; 5] = [1, 8, 64, 512, 4096];

    #[test]
    fn dp_assigns_every_tile_at_every_workload_point() {
        let params = llama_params("llama-3.1-8b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();
        let bounds = params.bounds.clone();

        let t0 = std::time::Instant::now();
        let workloads =
            solve(&fuf, &lib, &target, &inferred, &bounds, &WORKLOAD_POINTS).expect("solve");
        let elapsed_ms = t0.elapsed().as_millis();

        assert_eq!(workloads.per_num_tokens.len(), WORKLOAD_POINTS.len());
        for &m in &WORKLOAD_POINTS {
            let sfuf = workloads
                .per_num_tokens
                .get(&m)
                .unwrap_or_else(|| panic!("no sfuf for m={m}"));
            assert!(
                sfuf.is_cover_complete(fuf.len()),
                "cover not complete at m={m}"
            );
            // Starter library is single-op per impl, so subgraph
            // count equals tile count.
            assert_eq!(sfuf.num_subgraphs(), fuf.len(), "at m={m}");
            assert!(
                sfuf.predicted_us > 0.0 && sfuf.predicted_us.is_finite(),
                "predicted_us at m={m} = {}",
                sfuf.predicted_us,
            );
        }
        assert!(elapsed_ms < 100, "solve took {elapsed_ms} ms, budget 100");
    }

    #[test]
    fn gemm_cost_is_actually_counted() {
        // Regression: the prior greedy's gemm cost silently dropped
        // to zero because weight shapes weren't threaded. If this
        // regresses, prefill at M=4096 will collapse toward decode.
        let params = llama_params("llama-3.1-8b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();
        let bounds = params.bounds.clone();

        let workloads = solve(&fuf, &lib, &target, &inferred, &bounds, &[1, 4096]).unwrap();
        let decode_us = workloads.per_num_tokens[&1].predicted_us;
        let prefill_us = workloads.per_num_tokens[&4096].predicted_us;
        assert!(
            prefill_us > decode_us * 10.0,
            "prefill ({prefill_us}) should dwarf decode ({decode_us})",
        );
    }

    #[test]
    fn missing_impl_is_unclaimed_tile_error() {
        // Empty library — no Impl can match anything. First tile
        // should fail with UnclaimedTile, NOT be auto-claimed by
        // a fallback (the old ferrite DP had such a fallback for
        // ResidualAdd; we explicitly did not port it).
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build("hidden_states = embed(input_ids, embed_tokens);", &params);
        let lib = ImplementationLibrary::new();
        let target = l4_target();
        let bounds = params.bounds.clone();

        let err = solve(&fuf, &lib, &target, &inferred, &bounds, &[1]).unwrap_err();
        assert!(
            matches!(
                err,
                SolveError::UnclaimedTile {
                    op: OpKind::Embed,
                    ..
                }
            ),
            "expected UnclaimedTile on embed, got {err:?}",
        );
    }

    #[test]
    fn cost_fn_returning_none_is_fatal() {
        // First-None-fatal: no silent skipping. Even if another
        // candidate would cost fine, a None from any matched Impl
        // is a hard error.
        fn always_none(_ctx: &CostCtx) -> Option<f64> {
            None
        }

        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build("hidden_states = embed(input_ids, embed_tokens);", &params);
        let target = l4_target();

        let mut lib = ImplementationLibrary::new();
        lib.push(Implementation {
            name: "embed_buggy",
            op: OpKind::Embed,
            launch_kind: LaunchKind::HostCallable,
            workload_constraint: WorkloadConstraint::Any,
            target_filter: TargetFilter::Any,
            cost_fn: always_none,
            matches_fn: None,
            weight_layouts: &[Layout::Plain],
        });

        let err = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1]).unwrap_err();
        assert!(
            matches!(err, SolveError::UnreachableCost { .. }),
            "expected UnreachableCost, got {err:?}",
        );
    }

    #[test]
    fn workload_constraint_excludes_impls_outside_range() {
        // An Impl that only accepts M ≤ 8. At M=4096 it's excluded
        // by workload_constraint. If no other Impl covers embed,
        // solver must emit UnclaimedTile — not silently use the
        // excluded one.
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build("hidden_states = embed(input_ids, embed_tokens);", &params);
        let target = l4_target();

        fn cheap(_ctx: &CostCtx) -> Option<f64> {
            Some(1.0)
        }

        let mut lib = ImplementationLibrary::new();
        lib.push(Implementation {
            name: "embed_decode_only",
            op: OpKind::Embed,
            launch_kind: LaunchKind::HostCallable,
            workload_constraint: WorkloadConstraint::NumTokensRange { min: 1, max: 8 },
            target_filter: TargetFilter::Any,
            cost_fn: cheap,
            matches_fn: None,
            weight_layouts: &[Layout::Plain],
        });

        // At M=1 it's fine.
        let ok = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1]).unwrap();
        assert!(ok.per_num_tokens[&1].is_cover_complete(fuf.len()));

        // At M=4096 it's excluded; no other impl; UnclaimedTile.
        let err = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[4096]).unwrap_err();
        assert!(
            matches!(
                err,
                SolveError::UnclaimedTile {
                    op: OpKind::Embed,
                    ..
                }
            ),
            "expected UnclaimedTile, got {err:?}",
        );
    }

    #[test]
    fn multi_tile_matcher_collapses_to_one_subgraph() {
        // A synthetic multi-tile matcher: claims ANY two adjacent
        // add tiles (or returns None). Exercises the cover/impls
        // split so multiple tiles → one subgraph is proven wired.
        fn matches_double_add(seed: TileId, fuf: &Fuf) -> Option<MatchInfo> {
            if fuf.get(seed).op != OpKind::Add {
                return None;
            }
            // Find the next add after seed in FUF order, if any.
            let after = (seed.0 as usize + 1..fuf.len())
                .map(|i| TileId(i as u32))
                .find(|t| fuf.get(*t).op == OpKind::Add)?;
            Some(MatchInfo {
                claimed_tiles: vec![seed, after],
            })
        }

        // Cheaper than two single-add costs combined, so greedy
        // prefers this multi-tile match when it applies.
        fn cheap(_ctx: &CostCtx) -> Option<f64> {
            Some(0.001)
        }

        let params = llama_params("llama-3.2-1b");
        // Body with multiple adds so the matcher has something to claim.
        let (fuf, inferred) = build(
            r#"
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..2 {
                hidden_states = add(hidden_states, hidden_states);
                hidden_states = add(hidden_states, hidden_states);
            }
            "#,
            &params,
        );
        let target = l4_target();

        let mut lib = starter_library();
        // Append a "fused double add" that claims 2 adds in one
        // subgraph. Cheaper than two single adds from starter_library.
        lib.push(Implementation {
            name: "double_add_ref",
            op: OpKind::Add,
            launch_kind: LaunchKind::DeviceCallable,
            workload_constraint: WorkloadConstraint::Any,
            target_filter: TargetFilter::Any,
            cost_fn: cheap,
            matches_fn: Some(matches_double_add),
            // Add has no weight args; the double-add Impl
            // consumes two tile outputs and produces one tile output.
            weight_layouts: &[],
        });

        let workloads = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1]).unwrap();
        let sfuf = &workloads.per_num_tokens[&1];

        // 1 embed + 4 adds = 5 tiles, but the fused double-add
        // claims pairs of adds. Expected subgraph count: 1 embed +
        // 2 fused = 3 (each fused subgraph covers 2 tiles).
        assert_eq!(sfuf.num_subgraphs(), 3, "adds should pair into subgraphs");
        assert!(sfuf.is_cover_complete(fuf.len()));

        // Verify each fused subgraph actually covers 2 tiles.
        let add_subgraphs: Vec<_> = sfuf
            .subgraphs()
            .filter(|sg| {
                let imp_id = sfuf.impl_of(*sg).unwrap();
                lib.get(imp_id).op == OpKind::Add
            })
            .collect();
        assert_eq!(add_subgraphs.len(), 2);
        for sg in add_subgraphs {
            assert_eq!(sfuf.tiles_in_subgraph(sg).len(), 2);
        }
    }

    #[test]
    fn empty_workload_points_is_empty_result() {
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();
        let bounds = params.bounds.clone();

        let workloads = solve(&fuf, &lib, &target, &inferred, &bounds, &[]).unwrap();
        assert!(workloads.per_num_tokens.is_empty());
    }
}
