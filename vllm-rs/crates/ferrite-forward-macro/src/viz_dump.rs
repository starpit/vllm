// SPDX-License-Identifier: Apache-2.0
//! Env-gated JSON dump for the visualizer (`vllm-rs/viz`).
//!
//! When `FERRITE_VIZ_OUT` is set during `cargo build`, every
//! `#[forward]` invocation appends one `<arch>.json` file under that
//! directory containing the DSL source, the per-variant FUF, and the
//! per-workload solver output. The viz reads these as static assets.
//!
//! No serde derives — every type lives in compiler-internal modules.
//! We hand-build `serde_json::Value` trees so the schema can change
//! without touching every IR struct.
//!
//! Failure mode: any I/O error logs a `eprintln!` and returns; the
//! macro itself never fails because of viz dump.
//!
//! Concurrency: `cargo` builds multiple `#[forward]` invocations in
//! parallel. Each writes a *different* `<arch>.json` so file-level
//! collisions can't happen. Within one arch, multiple variants are
//! aggregated into one JSON inside this single call. We use
//! `OpenOptions::create_new` and ignore EEXIST — the first writer for
//! a given arch wins; reruns require deleting the dir.
//!
//! Stable output: BTreeMaps ordered by key, variants in solved order,
//! tile ids dense.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use quote::quote;
use serde_json::{Value, json};

use crate::classified::{BoolPred, Bound, Expr, ExternKind, LocalId, Program, Stmt, WeightId};
use crate::config::ModelParams;
use crate::fuf::{Fuf, FufInput};
use crate::impl_lib::{ImplId, ImplementationLibrary};
use crate::quantization::StorageFormat;
use crate::schedule::WorkloadLoops;
use crate::shape::{Dim, Shape};
use crate::solver::{Assignment, SubgraphId, WorkloadAssignments, WorkloadPoint};

/// The bundle of per-variant artifacts the macro pipeline already
/// computes; mirrors the inline `SolvedModel` in `lib.rs` but keeps
/// only what the viz needs.
pub struct VariantViz<'a> {
    pub model: &'a ModelParams,
    pub fuf: &'a Fuf,
    pub sfufs: &'a WorkloadAssignments,
    pub loops: &'a WorkloadLoops,
    /// Name of the canonical sibling whose generated forward fn this
    /// variant aliases. `None` for canonicals themselves. Non-canonicals
    /// emit a compact `{name, canonical}` stub — the FUF and workloads
    /// are byte-identical to the canonical's so duplicating them in the
    /// dump is pure bloat (Llama's 60-variant Cargo build became
    /// ~98 MB before this dedup).
    pub canonical: Option<String>,
}

/// Top-level entry. No-op unless `FERRITE_VIZ_OUT` is set.
pub fn maybe_dump(
    arch_name: &str,
    hf_arches: &[String],
    program: &Program,
    library: &ImplementationLibrary,
    variants: &[VariantViz<'_>],
    dsl_block: &syn::Block,
) {
    let Ok(out) = std::env::var("FERRITE_VIZ_OUT") else {
        return;
    };
    let out_dir = PathBuf::from(out);
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!(
            "ferrite-viz: create_dir_all({}) failed: {e}",
            out_dir.display()
        );
        return;
    }

    let dsl = render_block(dsl_block);

    let v = json!({
        "schema_version": 3,
        "arch": arch_name,
        "hf_arches": hf_arches,
        "dsl_source": dsl,
        "program": program_to_json(program),
        "variants": variants
            .iter()
            .map(|v| variant_to_json(v, program, library))
            .collect::<Vec<_>>(),
    });

    let path = out_dir.join(format!("{arch_name}.json"));
    let serialized = match serde_json::to_vec(&v) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ferrite-viz: serialize {arch_name} failed: {e}");
            return;
        }
    };
    // Atomic-ish write: tmp file + rename, so a concurrent reader
    // never sees a partial file.
    let tmp = out_dir.join(format!(".{arch_name}.json.tmp"));
    if let Err(e) = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&tmp)
        .and_then(|mut f| f.write_all(&serialized))
    {
        eprintln!("ferrite-viz: write {} failed: {e}", tmp.display());
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, &path) {
        eprintln!("ferrite-viz: rename {} failed: {e}", path.display());
        return;
    }
    eprintln!("ferrite-viz: wrote {}", path.display());
}

/// Render just the `forward!` body — not the surrounding `use` /
/// attrs / fn signature from the model crate's `src/lib.rs`. The body
/// alone is the math the user wrote; the rest is Rust glue.
///
/// Wrap the `Block` in a synthetic `fn _body() { … }`, prettyplease
/// it, then strip the wrapper lines and dedent. prettyplease takes a
/// `syn::File`, not a `Block` directly, so the wrapper is the
/// shortest path to formatted output.
fn render_block(block: &syn::Block) -> String {
    let file: syn::File = syn::parse_quote! {
        fn _body() #block
    };
    let pretty = prettyplease::unparse(&file);
    // prettyplease output:
    //   fn _body() {
    //       hidden_states = embed(input_ids, embed_tokens);
    //       ...
    //   }
    // Strip the first/last lines and dedent one level (4 spaces).
    let mut lines: Vec<&str> = pretty.lines().collect();
    if lines
        .first()
        .map(|l| l.trim_start().starts_with("fn _body"))
        .unwrap_or(false)
    {
        lines.remove(0);
    }
    if lines.last().map(|l| l.trim() == "}").unwrap_or(false) {
        lines.pop();
    }
    let dedented: Vec<String> = lines
        .iter()
        .map(|l| l.strip_prefix("    ").unwrap_or(l).to_string())
        .collect();
    let body = dedented.join("\n");
    // Token cleanup: prettyplease can't know our DSL is pseudo-Rust,
    // so it adds parens to bare statement-expressions like
    // `hidden_states = embed(...)`. Output is still readable, but
    // `let _ = ` may also appear. Leave both as-is; the viz reader
    // gets the structure regardless.
    let _ = quote! {}; // silence unused warning when feature flags shift
    body
}

fn variant_to_json(
    v: &VariantViz<'_>,
    program: &Program,
    library: &ImplementationLibrary,
) -> Value {
    let mut base = json!({
        "name": v.model.name,
        "source_stem": v.model.source_stem,
        "tie_word_embeddings": v.model.tie_word_embeddings,
        "bounds": v
            .model
            .bounds
            .iter()
            .map(|(k, v)| (k.clone(), json!(v)))
            .collect::<serde_json::Map<_, _>>(),
        "scalars": v
            .model
            .scalars
            .iter()
            .map(|(k, v)| (k.clone(), json!(v)))
            .collect::<serde_json::Map<_, _>>(),
    });
    let obj = base.as_object_mut().expect("base built as object");
    if let Some(c) = &v.canonical {
        obj.insert("canonical".into(), Value::String(c.clone()));
    } else {
        obj.insert("fuf".into(), fuf_to_json(v.fuf, program));
        obj.insert(
            "workloads".into(),
            workloads_to_json(v.fuf, v.sfufs, v.loops, library),
        );
    }
    base
}

fn fuf_to_json(fuf: &Fuf, program: &Program) -> Value {
    let nodes: Vec<Value> = fuf
        .nodes
        .iter()
        .map(|n| {
            json!({
                "id": n.id.0,
                "op": format!("{:?}", n.op),
                "inputs": n.inputs.iter().map(|i| input_to_json(i, program)).collect::<Vec<_>>(),
                "outputs": n.outputs.iter().map(shape_to_json).collect::<Vec<_>>(),
            })
        })
        .collect();
    json!({ "nodes": nodes })
}

fn input_to_json(i: &FufInput, program: &Program) -> Value {
    match i {
        FufInput::Tile { id, slot } => json!({
            "kind": "tile",
            "tile": id.0,
            "slot": slot,
        }),
        FufInput::Weight { id, index, storage } => json!({
            "kind": "weight",
            "name": weight_name(program, *id),
            "index": index,
            "storage": storage_label(storage),
        }),
        FufInput::Extern { kind, index } => json!({
            "kind": "extern",
            "name": extern_label(*kind),
            "index": index,
        }),
        FufInput::Scalar(v) => json!({ "kind": "scalar", "value": v }),
    }
}

fn shape_to_json(s: &Shape) -> Value {
    Value::Array(s.iter().map(dim_to_json).collect())
}

fn dim_to_json(d: &Dim) -> Value {
    match d {
        Dim::Lit(n) => json!(n),
        Dim::Bound(name) => Value::String(name.clone()),
        Dim::Mul(parts) => json!({ "mul": parts.iter().map(dim_to_json).collect::<Vec<_>>() }),
        Dim::Div(num, den) => json!({ "div": [dim_to_json(num), dim_to_json(den)] }),
        Dim::Var(_) => Value::String("?".into()),
    }
}

fn workloads_to_json(
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    loops: &WorkloadLoops,
    library: &ImplementationLibrary,
) -> Value {
    let mut out: Vec<Value> = Vec::new();
    for (wp, asn) in &sfufs.per_workload {
        let lp = loops.per_workload.get(wp);
        out.push(workload_to_json(*wp, asn, lp, fuf, library));
    }
    Value::Array(out)
}

fn workload_to_json(
    wp: WorkloadPoint,
    asn: &Assignment,
    lp: Option<&crate::schedule::Loop>,
    fuf: &Fuf,
    library: &ImplementationLibrary,
) -> Value {
    // Dense tile→subgraph vector indexed by TileId (so consumers
    // don't have to parse a string-keyed map).
    let mut tile_subgraph: Vec<Option<u32>> = vec![None; fuf.len()];
    for (tile, sg) in &asn.cover {
        tile_subgraph[tile.0 as usize] = Some(sg.0);
    }

    let mut subgraphs: Vec<(SubgraphId, ImplId)> =
        asn.impls.iter().map(|(sg, i)| (*sg, *i)).collect();
    subgraphs.sort_by_key(|(sg, _)| sg.0);
    let subgraph_impl: Vec<Value> = subgraphs
        .iter()
        .map(|(sg, i)| {
            json!({
                "sg": sg.0,
                "impl_id": i.0,
                "impl_name": library.get(*i).name(),
            })
        })
        .collect();

    let waves: Vec<Vec<u32>> = lp
        .map(|l| {
            l.waves
                .iter()
                .map(|w| w.subgraphs.iter().map(|(sg, _)| sg.0).collect())
                .collect()
        })
        .unwrap_or_default();

    json!({
        "num_tokens": wp.num_tokens,
        "sk_bucket": wp.sk_bucket,
        "predicted_us": asn.predicted_us,
        "num_subgraphs": asn.num_subgraphs(),
        "num_waves": waves.len(),
        "tile_subgraph": tile_subgraph,
        "subgraph_impl": subgraph_impl,
        "waves": waves,
    })
}

// ── stencil program emitter ──────────────────────────────────────
//
// Walks `Program::statements` and emits a *symbolic* tree —
// pre-unroll, one node per statement — designed for the viz's
// "stencil" mode (colored chips). Loops and ifs become container
// blocks; assigns become op-typed leaf chips.
//
// What this is NOT: the FUF (which is post-unroll, hundreds of
// nodes). This view stays small (~30 nodes per arch) and lines up
// 1:1 with the math the user wrote.

fn program_to_json(program: &Program) -> Value {
    json!({
        "blocks": stmt_list_to_json(&program.statements, program),
    })
}

/// Walk a statement list, dropping any synthesized-by-shape-inference
/// reshape stmts. `shape::infer` may rewrite the classified program
/// in-place to insert per-axis-factor reshape recoveries (Qwen3's
/// per-head q_norm / k_norm; Gemma3 similar). Those targets land in
/// `program.reshape_targets` and were never written by the user — the
/// math view should reflect the authored DSL, not the internal fixup.
/// The FUF view still shows them (they're real tiles in the unrolled
/// graph); only the stencil view filters.
fn stmt_list_to_json(stmts: &[Stmt], program: &Program) -> Value {
    let out: Vec<Value> = stmts
        .iter()
        .filter(|s| !is_synthesized_reshape(s, program))
        .map(|s| stmt_to_json(s, program))
        .collect();
    Value::Array(out)
}

fn is_synthesized_reshape(s: &Stmt, program: &Program) -> bool {
    match s {
        Stmt::Assign { target, .. } => program.reshape_targets.contains_key(target),
        // Tuple targets are never produced by reshape recovery, but
        // be conservative: only filter when *every* target was
        // synthesized.
        Stmt::AssignTuple { targets, .. } => targets
            .iter()
            .all(|t| program.reshape_targets.contains_key(t)),
        _ => false,
    }
}

fn stmt_to_json(s: &Stmt, program: &Program) -> Value {
    match s {
        Stmt::Assign { target, value } => assign_chip(&[*target], value, program),
        Stmt::AssignTuple { targets, value } => assign_chip(targets, value, program),
        Stmt::For {
            ivar,
            start,
            end,
            body,
            ..
        } => json!({
            "kind": "for",
            "ivar": local_name(program, *ivar),
            "start": bound_label(start),
            "end": bound_label(end),
            "body": stmt_list_to_json(body, program),
        }),
        Stmt::If {
            cond,
            then_body,
            else_body,
            ..
        } => json!({
            "kind": "if",
            "cond": pred_label(cond, program),
            "then": stmt_list_to_json(then_body, program),
            "else": stmt_list_to_json(else_body, program),
        }),
    }
}

/// One assign → an AST tree. The full `Expr` is serialized
/// recursively so the viz can render every nested call/binop/leaf as
/// its own colored block (an "AST silhouette"), not just the
/// outermost op.
fn assign_chip(targets: &[LocalId], value: &Expr, program: &Program) -> Value {
    let target_names: Vec<String> = targets.iter().map(|t| local_name(program, *t)).collect();
    json!({
        "kind": "assign",
        "targets": target_names,
        "value": expr_to_json(value, program),
    })
}

/// Recursive `Expr` serializer. Every node has a `kind`; composite
/// nodes (call / binop) carry `children`, leaves carry their own
/// fields. The viz renderer walks this tree and emits one nested
/// rectangle per node.
fn expr_to_json(e: &Expr, program: &Program) -> Value {
    match e {
        Expr::Call { op, args } => json!({
            "kind": "call",
            "op": format!("{op:?}"),
            "children": args.iter().map(|a| expr_to_json(a, program)).collect::<Vec<_>>(),
        }),
        Expr::Mul { lhs, rhs } => json!({
            "kind": "binop",
            "op": "*",
            "children": [expr_to_json(lhs, program), expr_to_json(rhs, program)],
        }),
        Expr::Add { lhs, rhs } => json!({
            "kind": "binop",
            "op": "+",
            "children": [expr_to_json(lhs, program), expr_to_json(rhs, program)],
        }),
        Expr::Local(id) => json!({ "kind": "local", "name": local_name(program, *id) }),
        Expr::Extern { kind, index } => json!({
            "kind": "extern",
            "name": format!("{kind:?}"),
            "index": index.map(|i| local_name(program, i)),
        }),
        Expr::Weight { id, index } => json!({
            "kind": "weight",
            "name": program.weights.path(*id).join("."),
            "index": index.map(|i| local_name(program, i)),
        }),
        Expr::ScalarLit(v) => json!({ "kind": "scalar", "value": v }),
        Expr::SqrtBound(name) => {
            json!({ "kind": "scalar_sym", "name": format!("sqrt({name})") })
        }
        Expr::ConfigScalar { name, recip } => json!({
            "kind": "scalar_sym",
            "name": if *recip { format!("1/{name}") } else { name.to_string() },
        }),
    }
}

fn local_name(program: &Program, id: LocalId) -> String {
    program.locals.name(id).to_string()
}

fn bound_label(b: &Bound) -> String {
    match b {
        Bound::Lit(n) => n.to_string(),
        Bound::Sym(name) => name.to_string(),
    }
}

fn pred_label(p: &BoolPred, program: &Program) -> String {
    match p {
        BoolPred::Modulo {
            ivar,
            divisor,
            remainder,
        } => format!(
            "{} % {} == {}",
            local_name(program, *ivar),
            bound_label(divisor),
            bound_label(remainder)
        ),
        BoolPred::NotModulo {
            ivar,
            divisor,
            remainder,
        } => format!(
            "{} % {} != {}",
            local_name(program, *ivar),
            bound_label(divisor),
            bound_label(remainder)
        ),
        BoolPred::Less { ivar, bound } => {
            format!("{} < {}", local_name(program, *ivar), bound_label(bound))
        }
    }
}

// ── label helpers ────────────────────────────────────────────────

fn extern_label(k: ExternKind) -> String {
    format!("{k:?}")
}

fn storage_label(s: &StorageFormat) -> String {
    format!("{s:?}")
}

/// Dotted-path name for a weight id (e.g. `"self_attn.q_proj"`),
/// derived from the interned segments in the program's weight table.
fn weight_name(program: &Program, id: WeightId) -> String {
    program.weights.path(id).join(".")
}

/// Re-export so `lib.rs` can talk to us without naming every member.
pub use self::call::dump_now;

mod call {
    use super::*;

    /// Bridge: the macro's `compile()` builds an inline `SolvedModel`
    /// vec; this helper takes the data it needs by parallel slices to
    /// avoid leaking the inline type definition into this module.
    /// `canonical_for[i]` is the canonical sibling's name for variant
    /// `i`; entries equal to `models[i].name` mean "this variant is
    /// itself canonical" → emit full FUF/workloads.
    #[allow(clippy::too_many_arguments)]
    pub fn dump_now<'a, M, F, S, L>(
        arch_name: &str,
        hf_arches: &[String],
        program: &Program,
        library: &ImplementationLibrary,
        dsl_block: &syn::Block,
        models: M,
        fufs: F,
        sfufss: S,
        loopss: L,
        canonical_for: &[String],
    ) where
        M: IntoIterator<Item = &'a ModelParams>,
        F: IntoIterator<Item = &'a Fuf>,
        S: IntoIterator<Item = &'a WorkloadAssignments>,
        L: IntoIterator<Item = &'a WorkloadLoops>,
    {
        // Skip the work entirely when not running under the viz env;
        // the per-variant aggregation below is cheap but the JSON
        // serialization isn't.
        if std::env::var_os("FERRITE_VIZ_OUT").is_none() {
            return;
        }
        let models: Vec<_> = models.into_iter().collect();
        let fufs: Vec<_> = fufs.into_iter().collect();
        let sfufss: Vec<_> = sfufss.into_iter().collect();
        let loopss: Vec<_> = loopss.into_iter().collect();
        let n = models.len();
        debug_assert_eq!(n, fufs.len());
        debug_assert_eq!(n, sfufss.len());
        debug_assert_eq!(n, loopss.len());
        debug_assert_eq!(n, canonical_for.len());
        let variants: Vec<VariantViz<'_>> = (0..n)
            .map(|i| {
                let canonical =
                    (canonical_for[i] != models[i].name).then(|| canonical_for[i].clone());
                VariantViz {
                    model: models[i],
                    fuf: fufs[i],
                    sfufs: sfufss[i],
                    loops: loopss[i],
                    canonical,
                }
            })
            .collect();
        super::maybe_dump(arch_name, hf_arches, program, library, &variants, dsl_block);
    }
}
