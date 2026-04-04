// SPDX-License-Identifier: Apache-2.0
//! ASCII diagram generator for the megakernel pipeline.
//!
//! Produces a human-readable visualization of the DAG showing:
//! - Op sequence with data flow edges
//! - Buffer shapes at each edge
//! - Barrier points
//! - Shared memory layout per op

use crate::dag::*;
use std::fmt::Write;

/// Generate an ASCII diagram of the megakernel pipeline.
pub fn render_diagram(dag: &ModelDag) -> String {
    let mut out = String::new();

    // Header
    writeln!(
        out,
        "╔══════════════════════════════════════════════════════════════╗"
    )
    .unwrap();
    writeln!(out, "║  megakernel! pipeline: {:<37} ║", dag.name).unwrap();
    writeln!(
        out,
        "╚══════════════════════════════════════════════════════════════╝"
    )
    .unwrap();
    writeln!(out).unwrap();

    // Params
    writeln!(out, "  Parameters:").unwrap();
    let mut param_keys: Vec<_> = dag.params.keys().collect();
    param_keys.sort();
    for key in &param_keys {
        writeln!(out, "    {key} = {}", dag.params[*key]).unwrap();
    }
    writeln!(out).unwrap();

    // Find layer loop bounds
    let first_loop_op = dag.ops.iter().position(|op| op.in_layer_loop);
    let last_loop_op = dag.ops.iter().rposition(|op| op.in_layer_loop);

    for (i, op) in dag.ops.iter().enumerate() {
        // Layer loop marker
        if Some(i) == first_loop_op {
            writeln!(
                out,
                "  ┌─── for layer in 0..NL ───────────────────────────────────┐"
            )
            .unwrap();
        }

        let prefix = if op.in_layer_loop { "  │  " } else { "  " };

        // Op box
        let (op_name, detail) = describe_op(&op.kind);
        writeln!(
            out,
            "{prefix}┌──────────────────────────────────────────────────────┐"
        )
        .unwrap();
        writeln!(out, "{prefix}│ [{i:>2}] {op_name:<49}│").unwrap();
        writeln!(out, "{prefix}│      {detail:<49}│").unwrap();
        writeln!(
            out,
            "{prefix}└──────────────────────────────────────────────────────┘"
        )
        .unwrap();

        // Show data flow edges
        for out_id in op.outputs() {
            if let Some(buf) = dag.buffers.get(out_id) {
                let consumers: Vec<_> = buf
                    .consumers
                    .iter()
                    .filter(|&&c| c != op.idx)
                    .map(|c| format!("op[{c}]"))
                    .collect();
                let dest = if consumers.is_empty() {
                    "→ (output)".to_string()
                } else {
                    format!("→ {}", consumers.join(", "))
                };
                writeln!(out, "{prefix}  ╰─ {}: {} {}", out_id, buf.shape, dest).unwrap();
            }
        }

        // Barrier indicator (between ops that cross SM boundaries)
        if i + 1 < dag.ops.len() && needs_barrier_between(dag, op, &dag.ops[i + 1]) {
            writeln!(
                out,
                "{prefix}  ── barrier ──────────────────────────────────────"
            )
            .unwrap();
        }

        // Close layer loop
        if Some(i) == last_loop_op {
            writeln!(
                out,
                "  └─────────────────────────────────────────────────────────┘"
            )
            .unwrap();
        }
    }

    // Shmem summary
    writeln!(out).unwrap();
    writeln!(out, "  Shared Memory Layout (sequential reuse):").unwrap();
    for op in &dag.ops {
        let shmem = crate::verify::estimate_op_shmem(dag, &op.kind);
        if shmem > 0 {
            let (name, _) = describe_op(&op.kind);
            writeln!(out, "    {name:<30} {shmem:>6} bytes").unwrap();
        }
    }

    out
}

fn describe_op(kind: &OpKind) -> (String, String) {
    match kind {
        OpKind::RmsNorm {
            input,
            weights,
            output,
        } => (
            format!("RmsNorm → {output}"),
            format!("{input} * {weights}"),
        ),
        OpKind::Gemm { a, b, output } => (format!("Gemm → {output}"), format!("{a} @ {b}^T")),
        OpKind::GemmAdd {
            a,
            b,
            residual,
            output,
        } => (
            format!("GemmAdd → {output}"),
            format!("{a} @ {b}^T + {residual}"),
        ),
        OpKind::RopeAppend {
            qkv,
            q_out,
            k_out,
            v_out,
            ..
        } => (
            format!("RopeAppend → {q_out},{k_out},{v_out}"),
            format!("split({qkv}) + RoPE + KV cache"),
        ),
        OpKind::AttentionDecode { q, output, .. } => (
            format!("AttentionDecode → {output}"),
            format!("FlashAttn({q}, kv_cache)"),
        ),
        OpKind::AttentionPrefill { q, output, .. } => (
            format!("AttentionPrefill → {output}"),
            format!("FlashAttnPrefill({q}, kv_cache)"),
        ),
        OpKind::Silu { input, output } => (
            format!("SiLU → {output}"),
            format!("{input} * sigmoid({input})"),
        ),
        OpKind::Mul { a, b, output } => (format!("Mul → {output}"), format!("{a} * {b}")),
    }
}

/// Heuristic: barrier needed when an op's output is consumed by a different op
/// that may run on different SMs.
fn needs_barrier_between(_dag: &ModelDag, producer: &Op, consumer: &Op) -> bool {
    // If consumer reads something the producer wrote, there's a data dependency.
    let produced: Vec<_> = producer.outputs().into_iter().cloned().collect();
    consumer.inputs().iter().any(|inp| produced.contains(inp))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_dag() -> ModelDag {
        let input: proc_macro2::TokenStream = quote::quote! {
            kernel llama_1b<NL=16, HD=2048, ID=5632, HDM=64, NAH=32, NKH=8, VS=128256> {
                for layer in 0..NL {
                    let normed = rmsnorm(hidden_states, attn_norm[layer]);
                    let qkv = gemm(normed, qkv_weights[layer]);
                    let (q, k, v) = rope_append(qkv, positions, kv_cache[layer]);
                    let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
                    hidden_states = gemm_add(attn, o_proj[layer], hidden_states);

                    let normed2 = rmsnorm(hidden_states, mlp_norm[layer]);
                    let gate = silu(gemm(normed2, gate_weights[layer]));
                    let up = gemm(normed2, up_weights[layer]);
                    hidden_states = gemm_add(gate * up, down_proj[layer], hidden_states);
                }
                let normed = rmsnorm(hidden_states, lm_head_norm);
                logits = gemm(normed, lm_head);
            }
        };
        let def: crate::parse::MegakernelDef = syn::parse2(input).unwrap();
        crate::parse::build_dag(&def).unwrap()
    }

    #[test]
    fn diagram_renders_without_panic() {
        let dag = build_test_dag();
        let diagram = render_diagram(&dag);
        assert!(diagram.contains("megakernel! pipeline: llama_1b"));
        assert!(diagram.contains("RmsNorm"));
        assert!(diagram.contains("Gemm"));
        assert!(diagram.contains("AttentionDecode"));
        assert!(diagram.contains("GemmAdd"));
        assert!(diagram.contains("SiLU"));
        assert!(diagram.contains("for layer in 0..NL"));
        // Print for visual inspection
        eprintln!("{diagram}");
    }
}
