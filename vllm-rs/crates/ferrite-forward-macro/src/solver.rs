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

    let ctx = CostCtx {
        fuf,
        profile: target,
        bounds,
    };

    for (i, node) in fuf.nodes.iter().enumerate() {
        for (imp_id, imp) in lib.iter_enumerated() {
            if !imp.target_compatible(target) {
                continue;
            }
            if !imp.workload_constraint().accepts(num_tokens as u32) {
                continue;
            }
            let Some(info) = imp.matches(fuf, node.id, target) else {
                continue;
            };
            let cost = imp.cost_us(&info, &ctx);
            if !cost.is_finite() {
                return Err(SolveError::UnreachableCost {
                    tile: node.id,
                    op: node.op,
                    impl_id: imp_id,
                    num_tokens,
                });
            }
            matches_at[i].push((imp_id, info, cost));
        }

        // Sort: claim size DESCENDING, then cost ASCENDING.
        //
        // Multi-tile claims represent fusion — by construction they
        // are preferable to covering the same tiles with separate
        // singletons (that is literally what fusion means: one
        // launch + shared memory traffic instead of N launches).
        // Picking the smaller claim at the seed can *orphan*
        // downstream tiles that only a multi-tile claim can cover
        // (e.g. Silu/Mul have no singleton impl — `silu_and_mul_fused`
        // is the only kernel). The greedy has to prefer the larger
        // claim to stay correct, not just optimal.
        //
        // Singleton-vs-singleton at the same seed falls through to
        // the cost tiebreak.
        matches_at[i].sort_by(|a, b| {
            b.1.claimed_tiles
                .len()
                .cmp(&a.1.claimed_tiles.len())
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
        });
    }
    let _ = inferred; // no longer used here — shapes come via ctx.fuf

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
        CostCtx, Handoff, Implementation, ImplementationLibrary, LaunchKind, Layout, MatchInfo,
        Resources, WorkloadConstraint, starter_library,
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

    /// Realistic Llama body with the full SwiGLU MLP (gate+silu+up+mul
    /// → down). The SwiGLU quadruple is the exemplar multi-tile claim
    /// exercised by `FusedGateUpSiluMulImpl`; if you strip silu/mul
    /// from this body the solver falls back to all-singleton coverage
    /// and the fusion tests below go dead.
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
            // Fusion savings per layer:
            //   SwiGLU (gate, up, silu, mul) → 1 subgraph (saves 3)
            //   attn-residual Add + post_attn_layernorm → 1 (saves 1)
            //   MLP-residual Add + next-layer input_layernorm (or
            //     final norm, on the last layer) → 1 (saves 1)
            // Total: 5 fewer subgraphs per layer. The first layer's
            // input_layernorm has no preceding Add (its upstream is
            // embed), so it stays a singleton — already accounted for
            // by the 5*NL count because the LAST Add's fusion partner
            // is the post-loop final RmsNorm.
            let nl = params.bounds["num_hidden_layers"] as usize;
            assert_eq!(
                sfuf.num_subgraphs(),
                fuf.len() - 5 * nl,
                "expected 5*NL fewer subgraphs than tiles at m={m} \
                 (SwiGLU + attn-residual + mlp-residual fusions)"
            );
            assert!(
                sfuf.predicted_us > 0.0 && sfuf.predicted_us.is_finite(),
                "predicted_us at m={m} = {}",
                sfuf.predicted_us,
            );
        }
        assert!(elapsed_ms < 100, "solve took {elapsed_ms} ms, budget 100");
    }

    #[test]
    fn swiglu_mlp_claimed_as_fused_subgraph_per_layer() {
        // For every unrolled MLP in a real Llama body, the (gate_gemm,
        // up_gemm, silu, mul) quadruple must collapse into one
        // FusedGateUpSiluMulImpl subgraph. No phase tags, no weight
        // names — this is the structural-matcher acid test.
        let params = llama_params("llama-3.1-8b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();

        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512, 4096],
        )
        .unwrap();

        let nl = params.bounds["num_hidden_layers"] as usize;

        for (&m, sfuf) in workloads.per_num_tokens.iter() {
            // Count subgraphs that claim 4 tiles of kinds {Gemm,
            // Gemm, Silu, Mul} with shared activation on the Gemms.
            let mut fused_count = 0;
            for sg in sfuf.subgraphs() {
                let tiles = sfuf.tiles_in_subgraph(sg);
                if tiles.len() != 4 {
                    continue;
                }
                let ops: Vec<OpKind> = tiles.iter().map(|t| fuf.get(*t).op).collect();
                let n_gemm = ops.iter().filter(|o| **o == OpKind::Gemm).count();
                let n_silu = ops.iter().filter(|o| **o == OpKind::Silu).count();
                let n_mul = ops.iter().filter(|o| **o == OpKind::Mul).count();
                if n_gemm == 2 && n_silu == 1 && n_mul == 1 {
                    fused_count += 1;
                    // And the bound Impl must be the fused one —
                    // lookup by name to avoid coupling to ImplId ordering.
                    let imp_id = sfuf.impl_of(sg).unwrap();
                    assert_eq!(
                        lib.get(imp_id).name(),
                        "fused_gate_up_silu_mul",
                        "subgraph at m={m} has fused topology but wrong impl",
                    );
                }
            }
            assert_eq!(
                fused_count, nl,
                "expected one fused SwiGLU subgraph per layer at m={m}",
            );
        }
    }

    #[test]
    fn add_rmsnorm_pairs_claimed_as_fused_subgraph() {
        // Every Add in a realistic Llama body has an immediate
        // RmsNorm consumer. FusedAddRmsNormImpl claims each pair as
        // one 2-tile subgraph. The first layer's input_layernorm is
        // the exception — its upstream is `embed`, not an Add — so
        // it stays a singleton RmsNormRefImpl claim.
        let params = llama_params("llama-3.1-8b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();

        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512, 4096],
        )
        .unwrap();

        let nl = params.bounds["num_hidden_layers"] as usize;

        for (&m, sfuf) in workloads.per_num_tokens.iter() {
            let mut fused_pair_count = 0;
            let mut singleton_rmsnorm_count = 0;
            for sg in sfuf.subgraphs() {
                let tiles = sfuf.tiles_in_subgraph(sg);
                let ops: Vec<OpKind> = tiles.iter().map(|t| fuf.get(*t).op).collect();
                if tiles.len() == 2 && ops.contains(&OpKind::Add) && ops.contains(&OpKind::RmsNorm)
                {
                    fused_pair_count += 1;
                    let imp_id = sfuf.impl_of(sg).unwrap();
                    assert_eq!(
                        lib.get(imp_id).name(),
                        "fused_add_rms_norm",
                        "subgraph at m={m} has Add+RmsNorm topology but wrong impl",
                    );
                } else if tiles.len() == 1 && ops[0] == OpKind::RmsNorm {
                    singleton_rmsnorm_count += 1;
                    let imp_id = sfuf.impl_of(sg).unwrap();
                    assert_eq!(
                        lib.get(imp_id).name(),
                        "rmsnorm_ref",
                        "singleton RmsNorm at m={m} bound to wrong impl",
                    );
                }
            }
            // 2 Adds per layer, each fuses with a following RmsNorm
            // (post_attn or next input_layernorm / final norm) → 2*NL
            // fused pairs.
            assert_eq!(
                fused_pair_count,
                2 * nl,
                "expected 2*NL fused Add+RmsNorm pairs at m={m}",
            );
            // Only the first layer's input_layernorm escapes fusion
            // (upstream = embed, not Add). One singleton RmsNorm.
            assert_eq!(
                singleton_rmsnorm_count, 1,
                "expected exactly one singleton RmsNorm (first input_layernorm) at m={m}",
            );
        }
    }

    #[test]
    fn fused_weight_accessor_collides_to_single_declaration_across_buckets() {
        // A fused impl declares one accessor per claim. Across many
        // buckets that pick the same Impl, WeightBundle-trait
        // aggregation must dedupe — otherwise user gets N copies of
        // the same accessor. Regression-guards the SFUF-walking
        // emission in codegen::emit_weight_bundle_trait.
        use crate::impl_lib::default_required_weights;
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();
        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512, 4096],
        )
        .unwrap();

        // Pick a SwiGLU-fused subgraph in the first bucket, record
        // its declared accessor name, then verify the same name
        // shows up exactly once in every other bucket's SFUF too.
        let first_sfuf = workloads.per_num_tokens.values().next().unwrap();
        let first_sg = first_sfuf
            .subgraphs()
            .find(|sg| {
                let tiles = first_sfuf.tiles_in_subgraph(*sg);
                tiles.len() == 4
                    && tiles.iter().any(|t| fuf.get(*t).op == OpKind::Silu)
                    && tiles.iter().any(|t| fuf.get(*t).op == OpKind::Mul)
            })
            .expect("at least one fused SwiGLU claim exists");
        let first_claim = first_sfuf.tiles_in_subgraph(first_sg);
        let first_imp = lib.get(first_sfuf.impl_of(first_sg).unwrap());
        let decls = first_imp.required_weights(&first_claim, &fuf, &classify_program(LLAMA_BODY));
        assert_eq!(
            decls.len(),
            1,
            "fused impl declares exactly one accessor per claim"
        );
        assert_eq!(
            decls[0].source_weights.len(),
            2,
            "fused accessor covers two source weights (gate_proj + up_proj)"
        );
        // And the default still works for Embed (singleton).
        let embed_sg = first_sfuf
            .subgraphs()
            .find(|sg| {
                let tiles = first_sfuf.tiles_in_subgraph(*sg);
                tiles.len() == 1 && fuf.get(tiles[0]).op == OpKind::Embed
            })
            .unwrap();
        let embed_claim = first_sfuf.tiles_in_subgraph(embed_sg);
        let default_decls =
            default_required_weights(&embed_claim, &fuf, &classify_program(LLAMA_BODY));
        assert_eq!(default_decls.len(), 1);
        assert_eq!(default_decls[0].source_weights.len(), 1);
    }

    fn classify_program(src: &str) -> crate::classified::Program {
        let file: syn::File =
            syn::parse_str(&format!("fn _c() {{ {src} }}")).expect("parse carrier");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = crate::parse::parse_block(block).unwrap();
        crate::classify::classify(&ast).unwrap()
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

    // Minimal trait impl used by tests that exercise solver
    // semantics at the trait boundary. Configurable cost + workload
    // constraint + optional multi-tile matcher.
    #[derive(Debug)]
    struct TestImpl {
        name: &'static str,
        op: OpKind,
        cost: f64,
        workload: WorkloadConstraint,
        multi_tile: Option<fn(&Fuf, TileId) -> Option<MatchInfo>>,
    }
    impl Implementation for TestImpl {
        fn name(&self) -> &'static str {
            self.name
        }
        fn target_compatible(&self, _p: &TargetProfile) -> bool {
            true
        }
        fn workload_constraint(&self) -> WorkloadConstraint {
            self.workload
        }
        fn matches(&self, fuf: &Fuf, seed: TileId, _p: &TargetProfile) -> Option<MatchInfo> {
            if let Some(f) = self.multi_tile {
                f(fuf, seed)
            } else if fuf.get(seed).op == self.op {
                Some(MatchInfo {
                    claimed_tiles: vec![seed],
                    boundary_inputs: vec![],
                    boundary_outputs: vec![seed],
                })
            } else {
                None
            }
        }
        fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
            self.cost
        }
        fn resources(&self, _m: &MatchInfo) -> Resources {
            Resources::ZERO
        }
        fn launch_kind(&self) -> LaunchKind {
            LaunchKind::HostCallback
        }
        fn supported_input_handoffs(&self) -> &[Handoff] {
            &[Handoff::StreamOrder]
        }
        fn supported_output_handoffs(&self) -> &[Handoff] {
            &[Handoff::StreamOrder]
        }
        fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
            vec![Layout::Any; m.boundary_inputs.len()]
        }
        fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
            vec![Layout::Any; m.boundary_outputs.len()]
        }
    }

    #[test]
    fn infinite_cost_is_fatal() {
        // The solver's cost aggregator treats non-finite costs as
        // UnreachableCost. A bug in an Impl's cost_us (returning
        // INFINITY or NaN) surfaces loudly rather than silently
        // propagating.
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build("hidden_states = embed(input_ids, embed_tokens);", &params);
        let target = l4_target();

        let mut lib = ImplementationLibrary::new();
        lib.push(Box::new(TestImpl {
            name: "embed_buggy",
            op: OpKind::Embed,
            cost: f64::INFINITY,
            workload: WorkloadConstraint::Any,
            multi_tile: None,
        }));

        let err = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1]).unwrap_err();
        assert!(
            matches!(err, SolveError::UnreachableCost { .. }),
            "expected UnreachableCost, got {err:?}",
        );
    }

    #[test]
    fn workload_constraint_excludes_impls_outside_range() {
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build("hidden_states = embed(input_ids, embed_tokens);", &params);
        let target = l4_target();

        let mut lib = ImplementationLibrary::new();
        lib.push(Box::new(TestImpl {
            name: "embed_decode_only",
            op: OpKind::Embed,
            cost: 1.0,
            workload: WorkloadConstraint::NumTokensRange { min: 1, max: 8 },
            multi_tile: None,
        }));

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
        fn matches_double_add(fuf: &Fuf, seed: TileId) -> Option<MatchInfo> {
            if fuf.get(seed).op != OpKind::Add {
                return None;
            }
            let after = (seed.0 as usize + 1..fuf.len())
                .map(|i| TileId(i as u32))
                .find(|t| fuf.get(*t).op == OpKind::Add)?;
            Some(MatchInfo {
                claimed_tiles: vec![seed, after],
                boundary_inputs: vec![],
                boundary_outputs: vec![seed, after],
            })
        }

        let params = llama_params("llama-3.2-1b");
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
        lib.push(Box::new(TestImpl {
            name: "double_add_ref",
            op: OpKind::Add,
            cost: 0.001, // cheaper than two single adds
            workload: WorkloadConstraint::Any,
            multi_tile: Some(matches_double_add),
        }));

        let workloads = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1]).unwrap();
        let sfuf = &workloads.per_num_tokens[&1];

        // 1 embed + 4 adds; fused double-add claims pairs.
        // Expected: 1 embed + 2 fused = 3 subgraphs.
        assert_eq!(sfuf.num_subgraphs(), 3, "adds should pair into subgraphs");
        assert!(sfuf.is_cover_complete(fuf.len()));

        // Find the subgraphs that cover Add tiles (structural —
        // ask the FUF what op each subgraph's tiles have).
        let add_subgraphs: Vec<_> = sfuf
            .subgraphs()
            .filter(|sg| {
                sfuf.tiles_in_subgraph(*sg)
                    .iter()
                    .all(|t| fuf.get(*t).op == OpKind::Add)
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
