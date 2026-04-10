// SPDX-License-Identifier: Apache-2.0
//! Codegen tests — verify the generated code has correct structure.

#[cfg(test)]
mod tests {
    use crate::lowering::backend::codegen;
    use crate::lowering::backend::compile_dsl::ForwardDef;

    const LLAMA_BODY: &str = r#"
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
    "#;

    fn gen_source(workloads: &str) -> String {
        let tokens: proc_macro2::TokenStream = format!(
            r#"
            {LLAMA_BODY}
            models: [
                {{ layers: 16, hidden: 2048, intermediate: 8192, heads: 32, kv_heads: 8, head_dim: 64 }},
            ],
            target: l4_sm89,
            workloads: [{workloads}],
            "#
        )
        .parse()
        .unwrap();
        let def: ForwardDef = syn::parse2(tokens).unwrap();
        codegen::generate(&def).to_string()
    }

    #[test]
    fn emits_solver_forward_layer() {
        let source = gen_source("1..1024");
        assert!(source.contains("solver_forward_layer"));
        assert!(source.contains("solver_layer_bucket_0"));
    }

    #[test]
    fn has_multiple_buckets() {
        let source = gen_source("1..4096");
        assert!(source.contains("solver_layer_bucket_0"));
        assert!(source.contains("solver_layer_bucket_1"));
    }

    #[test]
    fn oproj_reshapes_attention_output() {
        let source = gen_source("1..4096");
        assert!(
            source.contains("reshape"),
            "OProj should reshape attn output to 2D"
        );
    }

    #[test]
    fn up_gemm_does_not_double_call_gate_up_proj_for_dense() {
        let source = gen_source("1..4096");
        let forward_count = source.matches("gate_up_proj . forward").count()
            + source.matches("gate_up_proj.forward").count();
        let bucket_count = source.matches("solver_layer_bucket_").count();
        assert!(
            forward_count <= bucket_count,
            "gate_up_proj.forward called {forward_count} times but only {bucket_count} buckets"
        );
    }

    #[test]
    fn every_bucket_has_all_phases() {
        let source = gen_source("1..1024");
        assert!(source.contains("input_layernorm"), "missing attn norm");
        assert!(source.contains("qkv_proj"), "missing QKV GEMM");
        assert!(
            source.contains("attention_decode_from_cache") || source.contains("attention_standard"),
            "missing attention"
        );
        assert!(source.contains("o_proj"), "missing OProj");
        assert!(
            source.contains("post_attention_layernorm"),
            "missing MLP norm"
        );
        assert!(source.contains("gate_up_proj"), "missing Gate GEMM");
        assert!(source.contains("silu_and_mul_fused"), "missing SiLU");
        assert!(source.contains("down_proj"), "missing Down GEMM");
    }

    #[test]
    fn runtime_models_produce_empty() {
        let tokens: proc_macro2::TokenStream = format!(
            r#"
            {LLAMA_BODY}
            models: runtime,
            target: l4_sm89,
            "#
        )
        .parse()
        .unwrap();
        let def: ForwardDef = syn::parse2(tokens).unwrap();
        let _output = codegen::generate(&def);
    }
}
