// SPDX-License-Identifier: Apache-2.0
//! Fully Unrolled Forward (FUF) — CFG + unroll → [`TileGraph`].
//!
//! This is the IR builder. It is a thin walk over the tagged
//! output of [`unroll::unroll_tagged`]:
//!
//!   - one DSL op = one or more `TileNode`s (the "bullshit lift
//!     passes" for `rope_append` / `gate * up` are preserved here
//!     for now, to be ripped out in a later phase);
//!   - `GemmQ` / `GemmK` / … classification comes from the weight
//!     argument's name (see [`classify_gemm`]);
//!   - the `layer` field on each `TileNode` comes from the
//!     iteration phase tagged by unroll.
//!
//! FUF = TileGraph. There is no parallel IR type living here.

use std::collections::HashMap;

use crate::cfg::{Cfg, Instr};
use crate::lowering::tile_graph::{ModelDims, TileGraph, TileId, TileKind, TileNode};
use crate::parse::{Arg, OpCall};
use crate::unroll::{LoopPhase, UnrollError, unroll_tagged};

/// Error produced while building the FUF tile graph.
#[derive(Debug)]
pub enum FufError {
    /// Propagated from the unroll pass.
    Unroll(UnrollError),
    /// Op name isn't one the DSL knows how to lower to tile kinds.
    UnknownOp { op: String },
    /// A `let (a, b, c) = rope_append(...)` had the wrong arity.
    RopeAppendArity { got: usize },
}

impl From<UnrollError> for FufError {
    fn from(e: UnrollError) -> Self {
        FufError::Unroll(e)
    }
}

/// Build a `TileGraph` directly from the DSL's CFG.
///
/// Equivalent to the legacy `TileGraph::build_fuf(build_dag(def),
/// dims)` but routed through the honest CFG + unroll pipeline.
pub fn build_fuf(cfg: &Cfg, dims: ModelDims) -> Result<TileGraph, FufError> {
    let tagged = unroll_tagged(cfg)?;

    // num_layers mirrors legacy behavior: the DSL's `NL` param.
    // If no NL is present (e.g. a loop-free DSL), default to 1.
    let num_layers = cfg
        .loop_bounds
        .get("NL")
        .copied()
        .map(|n| n as u16)
        .unwrap_or(1);

    let mut builder = Builder {
        nodes: Vec::new(),
        buf_to_tile: HashMap::new(),
        current_hidden: None,
        num_layers,
    };

    // Pre-loop phase needs a valid `current_hidden` fallback before
    // any producer has emitted into `hidden_states`. The legacy path
    // synthesizes a sentinel `ResidualAdd` with no deps for this
    // purpose when the DSL doesn't start with `embed`. Preserve that.
    let has_prelude_producer = tagged
        .iter()
        .any(|(instr, phase)| matches!(phase, LoopPhase::PreLoop) && defines_hidden_states(instr));
    if !has_prelude_producer {
        let sentinel = builder.push(TileKind::ResidualAdd, 0, Vec::new(), None);
        builder.current_hidden = Some(sentinel);
    }

    for (instr, phase) in &tagged {
        builder.lower(instr, *phase)?;
    }

    Ok(TileGraph {
        nodes: builder.nodes,
        num_layers,
        dims,
    })
}

fn defines_hidden_states(instr: &Instr) -> bool {
    match instr {
        Instr::Let { name, .. } => name == "hidden_states",
        Instr::Assign { target, .. } => target == "hidden_states",
        Instr::LetTuple { names, .. } => names.iter().any(|n| n == "hidden_states"),
    }
}

// ── Builder ───────────────────────────────────────────────────────

struct Builder {
    nodes: Vec<TileNode>,
    /// Latest producing TileId for each DSL variable name. Gets
    /// overwritten on each `let`/`=` — straight-line SSA.
    buf_to_tile: HashMap<String, TileId>,
    /// Fallback producer for `hidden_states` when a dep lookup
    /// misses (e.g. the very first op reads `hidden_states` before
    /// anything has produced it in the unrolled stream). Updated
    /// whenever an op writes `hidden_states`.
    current_hidden: Option<TileId>,
    num_layers: u16,
}

impl Builder {
    fn push(
        &mut self,
        kind: TileKind,
        layer: u16,
        deps: Vec<TileId>,
        weight_name: Option<String>,
    ) -> TileId {
        let id = TileId(self.nodes.len() as u32);
        self.nodes.push(TileNode {
            id,
            kind,
            layer,
            deps,
            weight_name,
        });
        id
    }

    /// Layer tag for a tile based on the instruction's phase.
    /// Mirrors the legacy build_fuf layering:
    ///   - pre-loop → 0 (ops like `embed`)
    ///   - in-loop iter i → i
    ///   - post-loop → num_layers (final norm, lm_head)
    fn layer_for(&self, phase: LoopPhase) -> u16 {
        match phase {
            LoopPhase::PreLoop => 0,
            LoopPhase::InLoop { iter } => iter,
            LoopPhase::PostLoop => self.num_layers,
        }
    }

    /// Legacy `TileGraph::build_fuf` stamps pre-loop `Embed` with
    /// `layer=0` (not `PRE_LOOP_LAYER`). Mirror that for phase-1
    /// equivalence.
    fn layer_for_embed(&self, phase: LoopPhase) -> u16 {
        match phase {
            LoopPhase::PreLoop => 0,
            _ => self.layer_for(phase),
        }
    }

    /// Resolve an `Arg` to a producer `TileId` (or None if the arg
    /// is a true external like `input_ids`, `lm_head`, a weight,
    /// or a positions/rotary table).
    fn resolve_arg(&mut self, arg: &Arg, phase: LoopPhase) -> Result<Option<TileId>, FufError> {
        Ok(match arg {
            Arg::Var(name, None) => self.buf_to_tile.get(name.as_str()).copied().or_else(|| {
                if name == "hidden_states" {
                    self.current_hidden
                } else {
                    None
                }
            }),
            Arg::Var(_, Some(_)) => {
                // Symbolic indexed refs don't survive unroll.
                unreachable!("Arg::Var(Some(idx)) reached fuf builder — unroll bug")
            }
            Arg::VarAt(_, _) => {
                // External indexed buffer — `w[3]`, `self_attn.q_proj[3]`,
                // `kv_cache[3]`. Not a tile producer.
                None
            }
            Arg::Call(call) => {
                // Nested op call: emit its tile(s), return the output tile.
                Some(self.emit_call(call, phase)?)
            }
            Arg::Mul(a, b) => {
                // Honest lowering of `a * b`: one `Mul` tile with
                // two operand deps. Any kernel that wants packed
                // or fused semantics (e.g. CUTLASS's gate|up
                // silu_mul, vllm-rs's silu_and_mul_fused) declares
                // it claims `{Silu, Mul}` or the wider subgraph and
                // handles packing/fusion internally.
                let a_tile = self
                    .resolve_arg(a, phase)?
                    .or(self.current_hidden)
                    .expect("Arg::Mul operand has no producer");
                let b_tile = self
                    .resolve_arg(b, phase)?
                    .or(self.current_hidden)
                    .expect("Arg::Mul operand has no producer");
                let layer = self.layer_for(phase);
                let mul = self.push(TileKind::Mul, layer, vec![a_tile, b_tile], None);
                Some(mul)
            }
        })
    }

    /// Produce the external string name for a weight-like arg
    /// (`Arg::Var("lm_head", None)` → `"lm_head"`; `Arg::VarAt("w",
    /// 3)` → `"w"`). Used for `weight_name` on tiles and for
    /// `classify_gemm`. We drop the index because the legacy path
    /// stored the un-indexed buffer name and codegen appends
    /// `[layer]` at emit time.
    fn weight_string(arg: &Arg) -> Option<String> {
        match arg {
            Arg::Var(name, _) => Some(name.clone()),
            Arg::VarAt(name, _) => Some(name.clone()),
            _ => None,
        }
    }

    /// Emit an op call as one or more tiles and return the output
    /// tile (the one subsequent consumers should dep on).
    fn emit_call(&mut self, call: &OpCall, phase: LoopPhase) -> Result<TileId, FufError> {
        let op = call.op.to_string();
        let args = &call.args;

        // Resolve args up front so nested calls emit their tiles first.
        let resolved: Vec<Option<TileId>> = args
            .iter()
            .map(|a| self.resolve_arg(a, phase))
            .collect::<Result<_, _>>()?;

        match op.as_str() {
            "embed" => {
                let layer = self.layer_for_embed(phase);
                let weights = args.get(1).and_then(Self::weight_string);
                let tile = self.push(TileKind::Embed, layer, Vec::new(), weights);
                Ok(tile)
            }

            "rmsnorm" => {
                let layer = self.layer_for(phase);
                let dep = resolved[0].or(self.current_hidden).expect("rmsnorm input");
                let weights = args.get(1).and_then(Self::weight_string);
                let tile = self.push(TileKind::RmsNorm, layer, vec![dep], weights);
                Ok(tile)
            }

            "gemm" => {
                let layer = self.layer_for(phase);
                let dep = resolved[0].or(self.current_hidden).expect("gemm input");
                let weights = args.get(1).and_then(Self::weight_string);
                let kind = classify_gemm(weights.as_deref().unwrap_or(""));
                let tile = self.push(kind, layer, vec![dep], weights);
                Ok(tile)
            }

            "silu" => {
                // Honest lowering of `silu(x)`: one `Silu` tile
                // with a single input dep. Any kernel that wants
                // to fuse silu with a surrounding op (gate gemm +
                // silu + mul, silu+mul, etc.) declares it claims
                // the subgraph and fuses internally.
                let layer = self.layer_for(phase);
                let dep = resolved[0].or(self.current_hidden).expect("silu input");
                let tile = self.push(TileKind::Silu, layer, vec![dep], None);
                Ok(tile)
            }

            "bias_add" => {
                let layer = self.layer_for(phase);
                let dep = resolved[0].or(self.current_hidden).expect("bias_add input");
                let tile = self.push(TileKind::BiasAdd, layer, vec![dep], None);
                Ok(tile)
            }

            "add" => {
                let layer = self.layer_for(phase);
                let a = resolved[0].or(self.current_hidden).expect("add lhs");
                let b = resolved[1].or(self.current_hidden).expect("add rhs");
                let tile = self.push(TileKind::ResidualAdd, layer, vec![a, b], None);
                Ok(tile)
            }

            "attention" | "attention_decode" | "attention_prefill" => {
                let layer = self.layer_for(phase);
                let q_dep = resolved[0].or(self.current_hidden).expect("attention q");
                let kvw_dep = self
                    .nodes
                    .iter()
                    .rev()
                    .find(|n| n.kind == TileKind::KvCacheWrite && n.layer == layer)
                    .map(|n| n.id);
                let mut deps = vec![q_dep];
                if let Some(kvw) = kvw_dep {
                    deps.push(kvw);
                }
                let tile = self.push(TileKind::Attention, layer, deps, None);
                Ok(tile)
            }

            // `rope_append` is only ever bound via `let (q, k, v) = ...`
            // so its output binding is handled in `lower` below. If
            // someone writes it as a bare expression / single-let,
            // that's a DSL error — not our concern here.
            "rope_append" => Err(FufError::UnknownOp {
                op: "rope_append appeared outside a let-tuple binding".into(),
            }),

            other => Err(FufError::UnknownOp { op: other.into() }),
        }
    }

    /// Emit one top-level `Instr` and bind its outputs.
    fn lower(&mut self, instr: &Instr, phase: LoopPhase) -> Result<(), FufError> {
        match instr {
            Instr::Let { name, call } => {
                let tile = self.emit_call(call, phase)?;
                let name_s = name.to_string();
                self.buf_to_tile.insert(name_s.clone(), tile);
                if name_s == "hidden_states" {
                    self.current_hidden = Some(tile);
                }
                Ok(())
            }
            Instr::Assign { target, call } => {
                let tile = self.emit_call(call, phase)?;
                let target_s = target.to_string();
                self.buf_to_tile.insert(target_s.clone(), tile);
                if target_s == "hidden_states" {
                    self.current_hidden = Some(tile);
                }
                Ok(())
            }
            Instr::LetTuple { names, call } => {
                // `let (q, k, v) = rope_append(...)` is the only
                // tuple-producing op today. Lower it directly so we
                // can bind the three names to the three internal
                // tiles (Rope for q, KvCacheWrite for k/v) exactly
                // like the legacy build_fuf.
                let op = call.op.to_string();
                if op == "rope_append" {
                    self.lower_rope_append(names, &call.args, phase)?;
                    Ok(())
                } else {
                    Err(FufError::UnknownOp {
                        op: format!("let-tuple of op `{op}` not supported"),
                    })
                }
            }
        }
    }

    fn lower_rope_append(
        &mut self,
        names: &[syn::Ident],
        args: &[Arg],
        phase: LoopPhase,
    ) -> Result<(), FufError> {
        if names.len() != 3 {
            return Err(FufError::RopeAppendArity { got: names.len() });
        }

        let layer = self.layer_for(phase);
        // rope_append(q_in, k_in, v_in, positions, rotary, kv_cache[layer])
        let q_in = self
            .resolve_arg(&args[0], phase)?
            .or(self.current_hidden)
            .expect("rope_append q_in");
        let k_in = self
            .resolve_arg(&args[1], phase)?
            .or(self.current_hidden)
            .expect("rope_append k_in");
        let v_in = self
            .resolve_arg(&args[2], phase)?
            .or(self.current_hidden)
            .expect("rope_append v_in");
        // positions, rotary, kv_cache[layer] are externals — not tile producers.

        let split = self.push(TileKind::QkvSplit, layer, vec![q_in, k_in, v_in], None);
        let rope = self.push(TileKind::Rope, layer, vec![split], None);
        let _kvw = self.push(TileKind::KvCacheWrite, layer, vec![rope], None);

        // Legacy quirk preserved for phase-1 compat: the Rope /
        // KvCacheWrite tiles exist in the graph, but q/k/v stay
        // bound to the *pre*-rope GEMM tiles. Attention picks up
        // its KvCacheWrite dep via a separate by-layer scan below.
        // This whole mess (QkvSplit, KvCacheWrite, the decoupling
        // between output names and producer tiles) is scheduled
        // for removal in phase 4, which will emit a single honest
        // `RopeAppend` tile whose outputs are the real q/k/v.
        let _ = names;
        Ok(())
    }
}

/// Walk the CFG and extract the weight fields needed for the
/// generated `Layer` and `Model` structs.
///
/// Walks the unrolled instruction stream and classifies weight
/// args by:
///   1. the op that consumes them (`embed` → `Embedding`,
///      `rmsnorm` → `RmsNorm`, `gemm` → `LinearLayer`,
///      `rope_append`'s rotary arg → `RotaryCache`);
///   2. per-layer vs global: `Arg::VarAt(_, _)` (indexed in the
///      DSL, e.g. `self_attn.q_proj[layer]`) is per-layer;
///      `Arg::Var(_, None)` (unindexed, e.g. `lm_head`, `rotary`)
///      is global.
///
/// The `.bias` suffix is skipped — bias buffers are loaded as
/// part of their parent LinearLayer and don't need a separate
/// struct field.
///
/// Returns `(per_layer_fields, global_fields)` sorted by name for
/// stable output.
pub fn extract_weight_fields(
    cfg: &Cfg,
) -> Result<
    (
        Vec<crate::lowering::backend::codegen::FieldSpec>,
        Vec<crate::lowering::backend::codegen::FieldSpec>,
    ),
    FufError,
> {
    use crate::lowering::backend::codegen::FieldSpec;
    use std::collections::BTreeMap;

    let tagged = unroll_tagged(cfg)?;

    // Per-weight-name classification. First-seen wins — a name that
    // appears first inside a loop body stays as per-layer even if
    // referenced later outside (extremely unusual; mirror the
    // legacy's first-consumer-wins behavior).
    let mut per_layer: BTreeMap<String, &'static str> = BTreeMap::new();
    let mut global: BTreeMap<String, &'static str> = BTreeMap::new();

    for (instr, _phase) in &tagged {
        let call = match instr {
            Instr::Let { call, .. } | Instr::LetTuple { call, .. } | Instr::Assign { call, .. } => {
                call
            }
        };
        collect_weight_args_from_call(call, &mut per_layer, &mut global);
    }

    let to_specs = |m: BTreeMap<String, &'static str>| -> Vec<FieldSpec> {
        m.into_iter()
            .map(|(name, ty)| FieldSpec { name, ty })
            .collect()
    };

    Ok((to_specs(per_layer), to_specs(global)))
}

/// Classify all weight-shaped args of one `OpCall`, recursing into
/// nested `Arg::Call` so `gemm(silu(gemm(x, w1)), w2)` picks up
/// both `w1` and `w2`.
fn collect_weight_args_from_call(
    call: &OpCall,
    per_layer: &mut std::collections::BTreeMap<String, &'static str>,
    global: &mut std::collections::BTreeMap<String, &'static str>,
) {
    let op = call.op.to_string();

    // Per-op weight-arg positions. None = no weights for this op.
    // (arg_index, rust_type)
    let weight_positions: &[(usize, &'static str)] = match op.as_str() {
        "embed" => &[(1, "Embedding")],
        "rmsnorm" => &[(1, "RmsNorm")],
        "gemm" => &[(1, "LinearLayer")],
        "bias_add" => &[], // bias is folded into parent LinearLayer
        "rope_append" => &[(4, "RotaryCache")],
        _ => &[],
    };

    for (arg_idx, ty) in weight_positions {
        if let Some(arg) = call.args.get(*arg_idx) {
            classify_weight_arg(arg, ty, per_layer, global);
        }
    }

    // Recurse into nested calls / muls so inner weight args are seen.
    for arg in &call.args {
        recurse_weight_args_in_arg(arg, per_layer, global);
    }
}

fn recurse_weight_args_in_arg(
    arg: &Arg,
    per_layer: &mut std::collections::BTreeMap<String, &'static str>,
    global: &mut std::collections::BTreeMap<String, &'static str>,
) {
    match arg {
        Arg::Call(c) => collect_weight_args_from_call(c, per_layer, global),
        Arg::Mul(a, b) => {
            recurse_weight_args_in_arg(a, per_layer, global);
            recurse_weight_args_in_arg(b, per_layer, global);
        }
        _ => {}
    }
}

fn classify_weight_arg(
    arg: &Arg,
    ty: &'static str,
    per_layer: &mut std::collections::BTreeMap<String, &'static str>,
    global: &mut std::collections::BTreeMap<String, &'static str>,
) {
    let (name, is_per_layer) = match arg {
        Arg::VarAt(n, _) => (n.clone(), true),
        Arg::Var(n, None) => (n.clone(), false),
        // Weight args shouldn't be nested calls / mul / symbolic-idx
        // in any DSL we support; ignore.
        _ => return,
    };
    if name.ends_with(".bias") {
        return;
    }
    if is_per_layer {
        per_layer.entry(name).or_insert(ty);
    } else {
        global.entry(name).or_insert(ty);
    }
}

/// Classify a GEMM tile by its weight argument's string name.
///
/// Same substring rules as the legacy `tile_graph::classify_gemm`,
/// duplicated here so fuf.rs doesn't depend on the legacy builder's
/// private helpers. Once phase 3 lands and the legacy path is
/// deleted, this becomes the only copy.
fn classify_gemm(weight_name: &str) -> TileKind {
    let w = weight_name.to_lowercase();
    if w.contains("q_proj") {
        TileKind::GemmQ
    } else if w.contains("k_proj") {
        TileKind::GemmK
    } else if w.contains("v_proj") {
        TileKind::GemmV
    } else if w.contains("o_proj") {
        TileKind::GemmOProj
    } else if w.contains("lm_head") {
        TileKind::GemmLmHead
    } else if w.contains("gate_proj") {
        TileKind::GemmGate
    } else if w.contains("up_proj") {
        TileKind::GemmUp
    } else if w.contains("down_proj") {
        TileKind::GemmDown
    } else {
        TileKind::GemmQ
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::build_cfg;
    use crate::parse::MegakernelDef;

    fn parse_dsl(src: &str) -> MegakernelDef {
        let tokens: proc_macro2::TokenStream = src.parse().unwrap();
        syn::parse2(tokens).unwrap()
    }

    const LLAMA_DSL: &str = r#"
        kernel llama<NL=2, HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..NL {
                let normed = rmsnorm(hidden_states, input_layernorm[layer]);
                let q = gemm(normed, self_attn.q_proj[layer]);
                let k = gemm(normed, self_attn.k_proj[layer]);
                let v = gemm(normed, self_attn.v_proj[layer]);
                let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                let attn = attention(q, k, v, kv_cache[layer], block_table);
                let oproj = gemm(attn, self_attn.o_proj[layer]);
                hidden_states = add(oproj, hidden_states);

                let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                let up = gemm(normed2, mlp.up_proj[layer]);
                let down = gemm(gate * up, mlp.down_proj[layer]);
                hidden_states = add(down, hidden_states);
            }
            let final_norm = rmsnorm(hidden_states, norm);
            logits = gemm(final_norm, lm_head);
        }
    "#;

    /// Golden: the CFG-driven FUF on the canonical LLaMA DSL
    /// produces exactly the kind-and-layer sequence downstream
    /// solver/codegen expect — 1 pre-loop Embed + 17 tiles per
    /// layer (matching the legacy per-layer tile layout, preserved
    /// for phase-1 compatibility) + 2 post-loop tiles (final RMS
    /// norm + lm_head).
    #[test]
    fn build_fuf_golden_llama_shape() {
        let def = parse_dsl(LLAMA_DSL);
        let cfg = build_cfg(&def);
        let tg = build_fuf(&cfg, ModelDims::LLAMA_3_2_1B).expect("build_fuf");

        // 1 embed + 17 per layer * 2 layers + 2 post-loop = 37.
        assert_eq!(tg.nodes.len(), 37, "tile count on 2-layer LLaMA");
        assert_eq!(tg.num_layers, 2);

        // Per-layer tile pattern (kind, layer) — tiles 1..=17 for
        // layer 0, tiles 18..=34 for layer 1.
        let expected_per_layer = [
            TileKind::RmsNorm, // attn_norm
            TileKind::GemmQ,
            TileKind::GemmK,
            TileKind::GemmV,
            TileKind::QkvSplit,     // phantom (phase 4c removes)
            TileKind::Rope,         // phantom
            TileKind::KvCacheWrite, // phantom
            TileKind::Attention,
            TileKind::GemmOProj,
            TileKind::ResidualAdd, // attn residual
            TileKind::RmsNorm,     // mlp_norm
            TileKind::GemmGate,
            TileKind::Silu, // honest silu(gate_gemm) — was passthrough before 4a
            TileKind::GemmUp,
            TileKind::Mul, // honest gate * up — was GateUpConcat+SiluMul before 4a
            TileKind::GemmDown,
            TileKind::ResidualAdd, // mlp residual
        ];
        assert_eq!(tg.nodes[0].kind, TileKind::Embed);
        for layer in 0..2u16 {
            let offset = 1 + (layer as usize) * expected_per_layer.len();
            for (i, kind) in expected_per_layer.iter().enumerate() {
                let n = &tg.nodes[offset + i];
                assert_eq!(n.kind, *kind, "tile {} kind (layer {layer})", offset + i);
                assert_eq!(
                    n.layer,
                    layer,
                    "tile {} layer (expected {layer})",
                    offset + i
                );
            }
        }

        // Post-loop: final RMS norm at layer == num_layers, then
        // GemmLmHead.
        let post0 = &tg.nodes[35];
        let post1 = &tg.nodes[36];
        assert_eq!(post0.kind, TileKind::RmsNorm);
        assert_eq!(post0.layer, 2);
        assert_eq!(post0.weight_name.as_deref(), Some("norm"));
        assert_eq!(post1.kind, TileKind::GemmLmHead);
        assert_eq!(post1.layer, 2);
        assert_eq!(post1.weight_name.as_deref(), Some("lm_head"));
    }

    /// Loop-carried `hidden_states` should thread through every
    /// layer's residual adds — a property the legacy DAG pipeline
    /// broke (see commit log for `refactor(fuf): produce TileGraph
    /// directly`).
    #[test]
    fn build_fuf_cross_layer_residual_dep() {
        let def = parse_dsl(LLAMA_DSL);
        let cfg = build_cfg(&def);
        let tg = build_fuf(&cfg, ModelDims::LLAMA_3_2_1B).unwrap();

        // Layer 0's mlp residual add is at index 1 + 16 = 17.
        // Layer 1's attn rmsnorm is at index 18.
        let layer0_mlp_add = &tg.nodes[17];
        let layer1_attn_norm = &tg.nodes[18];
        assert_eq!(layer0_mlp_add.kind, TileKind::ResidualAdd);
        assert_eq!(layer1_attn_norm.kind, TileKind::RmsNorm);
        assert!(
            layer1_attn_norm.deps.contains(&layer0_mlp_add.id),
            "layer 1's attn norm should depend on layer 0's final \
             residual add ({:?}); got deps {:?}",
            layer0_mlp_add.id,
            layer1_attn_norm.deps,
        );
    }

    #[test]
    fn build_fuf_no_loops() {
        let def = parse_dsl(
            r#"
            kernel test<HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                logits = gemm(hidden_states, lm_head);
            }
            "#,
        );
        let cfg = build_cfg(&def);
        let tg = build_fuf(&cfg, ModelDims::LLAMA_3_2_1B).unwrap();

        // Expect exactly: Embed, GemmLmHead.
        assert_eq!(tg.nodes.len(), 2);
        assert_eq!(tg.nodes[0].kind, TileKind::Embed);
        assert_eq!(tg.nodes[1].kind, TileKind::GemmLmHead);
        assert_eq!(tg.nodes[1].weight_name.as_deref(), Some("lm_head"));
        assert_eq!(tg.nodes[1].deps, vec![tg.nodes[0].id]);
    }

    #[test]
    fn extract_weight_fields_llama() {
        use crate::lowering::backend::codegen::FieldSpec;
        let def = parse_dsl(LLAMA_DSL);
        let cfg = build_cfg(&def);

        let (per_layer, global) = extract_weight_fields(&cfg).expect("extract");

        let pl_names: Vec<_> = per_layer.iter().map(|f| f.name.clone()).collect();
        let g_names: Vec<_> = global.iter().map(|f| f.name.clone()).collect();

        // Per-layer: every HF weight path that's indexed by `[layer]`
        // in the DSL should show up.
        for expected in [
            "input_layernorm",
            "mlp.down_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "post_attention_layernorm",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "self_attn.q_proj",
            "self_attn.v_proj",
        ] {
            assert!(
                pl_names.contains(&expected.to_string()),
                "missing per-layer field {expected}: got {pl_names:?}",
            );
        }
        // Global: unindexed weights.
        for expected in ["embed_tokens", "lm_head", "norm", "rotary"] {
            assert!(
                g_names.contains(&expected.to_string()),
                "missing global field {expected}: got {g_names:?}",
            );
        }

        // Type classification.
        let ty_of = |v: &[FieldSpec], name: &str| -> &'static str {
            v.iter().find(|f| f.name == name).unwrap().ty
        };
        assert_eq!(ty_of(&per_layer, "self_attn.q_proj"), "LinearLayer");
        assert_eq!(ty_of(&per_layer, "input_layernorm"), "RmsNorm");
        assert_eq!(ty_of(&global, "embed_tokens"), "Embedding");
        assert_eq!(ty_of(&global, "rotary"), "RotaryCache");
        assert_eq!(ty_of(&global, "lm_head"), "LinearLayer");
        assert_eq!(ty_of(&global, "norm"), "RmsNorm");
    }

    /// Golden: the full-LLaMA DSL produces *exactly* the expected
    /// sorted per-layer and global field lists, by name and type.
    /// Regression guard for any drift in the extractor.
    #[test]
    fn extract_weight_fields_golden_llama() {
        let def = parse_dsl(LLAMA_DSL);
        let cfg = build_cfg(&def);

        let (per_layer, global) = extract_weight_fields(&cfg).expect("extract");

        let pl: Vec<(String, &'static str)> =
            per_layer.iter().map(|f| (f.name.clone(), f.ty)).collect();
        assert_eq!(
            pl,
            vec![
                ("input_layernorm".to_string(), "RmsNorm"),
                ("mlp.down_proj".to_string(), "LinearLayer"),
                ("mlp.gate_proj".to_string(), "LinearLayer"),
                ("mlp.up_proj".to_string(), "LinearLayer"),
                ("post_attention_layernorm".to_string(), "RmsNorm"),
                ("self_attn.k_proj".to_string(), "LinearLayer"),
                ("self_attn.o_proj".to_string(), "LinearLayer"),
                ("self_attn.q_proj".to_string(), "LinearLayer"),
                ("self_attn.v_proj".to_string(), "LinearLayer"),
            ],
        );

        let g: Vec<(String, &'static str)> =
            global.iter().map(|f| (f.name.clone(), f.ty)).collect();
        assert_eq!(
            g,
            vec![
                ("embed_tokens".to_string(), "Embedding"),
                ("lm_head".to_string(), "LinearLayer"),
                ("norm".to_string(), "RmsNorm"),
                ("rotary".to_string(), "RotaryCache"),
            ],
        );
    }

    #[test]
    fn extract_weight_fields_skips_bias() {
        // `*.bias` weights are handled by their parent LinearLayer's
        // loader and shouldn't appear as their own fields.
        let def = parse_dsl(
            r#"
            kernel test<NL=2, HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                for layer in 0..NL {
                    let n = rmsnorm(hidden_states, input_layernorm[layer]);
                    let q = gemm(n, self_attn.q_proj[layer]);
                    let q = bias_add(q, self_attn.q_proj.bias[layer]);
                    hidden_states = add(q, hidden_states);
                }
                logits = gemm(hidden_states, lm_head);
            }
            "#,
        );
        let cfg = build_cfg(&def);
        let (per_layer, _global) = extract_weight_fields(&cfg).expect("extract");
        let names: Vec<_> = per_layer.iter().map(|f| f.name.clone()).collect();
        assert!(
            !names.iter().any(|n| n.ends_with(".bias")),
            "bias fields should be skipped: got {names:?}",
        );
        assert!(
            names.iter().any(|n| n == "self_attn.q_proj"),
            "parent LinearLayer field should be kept: got {names:?}",
        );
    }

    #[test]
    fn build_fuf_num_layers_from_nl() {
        let def = parse_dsl(
            r#"
            kernel test<NL=3, HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                for layer in 0..NL {
                    hidden_states = gemm(hidden_states, self_attn.q_proj[layer]);
                }
                logits = gemm(hidden_states, lm_head);
            }
            "#,
        );
        let cfg = build_cfg(&def);
        let tg = build_fuf(&cfg, ModelDims::LLAMA_3_2_1B).unwrap();
        assert_eq!(tg.num_layers, 3);
    }
}
