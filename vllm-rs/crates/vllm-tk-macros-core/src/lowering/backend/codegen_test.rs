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
            let (q, k, v) = rope_append(qkv, positions, rotary, kv_cache[layer]);
            let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
            hidden_states = gemm_add(attn, o_proj[layer], hidden_states);

            let normed2 = rmsnorm(hidden_states, mlp_norm[layer]);
            let gate = silu(gemm(normed2, gate_weights[layer]));
            let up = gemm(normed2, up_weights[layer]);
            hidden_states = gemm_add(gate * up, down_proj[layer], hidden_states);
        }
        logits = gemm(hidden_states, lm_head);
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
    fn emits_model_forward_and_bucket_fns() {
        // Generated shape: `impl Model { fn forward(&self, ...) }`
        // plus per-bucket fns named `solver_layer_bucket_N`.
        let source = gen_source("1..1024");
        assert!(source.contains("impl Model"), "missing impl Model block");
        assert!(
            source.contains("pub unsafe fn forward"),
            "missing Model::forward method"
        );
        assert!(source.contains("solver_layer_bucket_0"));
    }

    #[test]
    fn emits_layer_and_model_structs() {
        // Task 22: every `forward!()` expansion emits a `Layer`
        // struct (one field per per-layer weight in the DSL body) and
        // a `Model` struct (`layers: Vec<Layer>` + one field per
        // global weight). Pins the field set so a DSL body change
        // that accidentally drops or renames a weight gets caught
        // at cargo-test time.
        let source = gen_source("1..1024");

        assert!(
            source.contains("pub struct Layer"),
            "generated source missing Layer struct"
        );
        assert!(
            source.contains("pub struct Model"),
            "generated source missing Model struct"
        );

        // Every per-layer weight referenced by LLAMA_BODY becomes a
        // field on the Layer struct. The generator emits tokens
        // as `pub attn_norm : RmsNorm , ...` with spaces around `:`.
        // `up_weights` is special-cased to `Option<LinearLayer>` to
        // preserve the legacy fused-gate_up runtime behavior — see
        // `field_token` in codegen.rs.
        for (name, ty) in [
            ("attn_norm", "RmsNorm"),
            ("mlp_norm", "RmsNorm"),
            ("qkv_weights", "LinearLayer"),
            ("o_proj", "LinearLayer"),
            ("gate_weights", "LinearLayer"),
            ("down_proj", "LinearLayer"),
        ] {
            let field_tokens = format!("pub {name} : {ty}");
            assert!(
                source.contains(&field_tokens),
                "Layer struct missing field `{field_tokens}`"
            );
        }
        assert!(
            source.contains("pub up_weights : Option < LinearLayer >"),
            "Layer struct missing `up_weights: Option<LinearLayer>` (Llama fused gate_up holdover)"
        );

        // Global weights live on the Model struct. LLAMA_BODY today
        // has only `lm_head` as a global weight — the rest
        // (`embed_tokens`, `rotary`, `final_norm`) land once the
        // body is extended in Task 24.
        assert!(
            source.contains("pub layers : Vec < Layer >"),
            "Model struct missing `layers: Vec<Layer>` field"
        );
        assert!(
            source.contains("pub lm_head : LinearLayer"),
            "Model struct missing `lm_head: LinearLayer` field"
        );
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
    fn up_gemm_does_not_double_call_gate_weights_forward_for_dense() {
        // Layer struct exposes `gate_weights` (which the loader sets
        // to the fused gate_up LinearLayer on Llama-family models)
        // and `up_weights: Option<LinearLayer>` (None in the fused
        // case). Each bucket function should call `gate_weights` at
        // most once per bucket — the Up phase's emission is gated on
        // `layer.up_weights.is_some()` which falls through on Llama.
        let source = gen_source("1..4096");
        let forward_count = source.matches("gate_weights . forward").count()
            + source.matches("gate_weights.forward").count();
        let bucket_count = source.matches("solver_layer_bucket_").count();
        assert!(
            forward_count <= bucket_count,
            "gate_weights.forward called {forward_count} times but only {bucket_count} buckets"
        );
    }

    #[test]
    fn every_bucket_has_all_phases() {
        // The generated code touches each decoder-layer phase at
        // least once per bucket. Substrings match the new direct
        // field access (`layer.attn_norm`, `layer.qkv_weights`, …),
        // not the retired `layer.self_attn.*` / `layer.mlp.*` paths.
        let source = gen_source("1..1024");
        assert!(source.contains("attn_norm"), "missing attn norm");
        assert!(source.contains("qkv_weights"), "missing QKV GEMM");
        assert!(
            source.contains("attention_decode_from_cache") || source.contains("attention_standard"),
            "missing attention"
        );
        assert!(source.contains("o_proj"), "missing OProj");
        assert!(source.contains("mlp_norm"), "missing MLP norm");
        assert!(source.contains("gate_weights"), "missing Gate GEMM");
        assert!(source.contains("silu_and_mul_fused"), "missing SiLU");
        assert!(source.contains("down_proj"), "missing Down GEMM");
    }

    #[test]
    #[ignore]
    fn solve_time_one_forward() {
        // Time one full forward! macro expansion for Llama 3.2 3B.
        // Run with:
        //   cargo test --release -p vllm-tk-macros-core -- --ignored solve_time_one_forward --nocapture
        let start = std::time::Instant::now();
        let source = gen_source("1..4096");
        let elapsed = start.elapsed();
        println!(
            "solve_time_one_forward: {:?} ({} bytes)",
            elapsed,
            source.len()
        );
    }

    // (The old `qwen2_*` structural tests targeted the retired
    // `name:` suffix on `forward!` and the legacy `OpKind::GemmBias`
    // DSL op. Bias coverage now flows through `ModelDims::qkv_bias`
    // + `from_model_dag`, and Qwen2 support arrives via a separate
    // `forward!()` invocation in `qwen2.rs`. Equivalent coverage
    // tests will be added once that invocation lands.)

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
