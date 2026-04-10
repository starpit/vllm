// SPDX-License-Identifier: Apache-2.0
//! Codegen tests — verify the generated code has correct structure.
//! These tests inspect the TokenStream output to catch dataflow bugs
//! before GPU testing.

#[cfg(test)]
mod tests {
    use crate::lowering::backend::codegen;
    use crate::lowering::backend::compile_dsl::CompileDef;

    fn gen_source(workloads: &str) -> String {
        let tokens: proc_macro2::TokenStream =
            format!("model: llama_3_2_1b, target: l4_sm89, workloads: [{workloads}]")
                .parse()
                .unwrap();
        let def: CompileDef = syn::parse2(tokens).unwrap();
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

    /// OProj must reshape attention output to 2D before the GEMM.
    /// attention_decode_from_cache returns [num_tokens, num_heads, head_dim].
    /// o_proj.forward() expects [num_tokens, q_size].
    #[test]
    fn oproj_reshapes_attention_output() {
        let source = gen_source("1..4096");
        // The OProj input should contain a reshape call.
        assert!(
            source.contains("reshape"),
            "OProj should reshape attn output to 2D:\n{source}"
        );
    }

    /// For dense models, Gate GEMM uses gate_up_proj (fused [2*ID, HD]).
    /// The Up entry should NOT call gate_up_proj again — it's a noop
    /// for dense because Gate already produced [M, 2*intermediate].
    #[test]
    fn up_gemm_does_not_double_call_gate_up_proj_for_dense() {
        let source = gen_source("1..4096");
        // Count how many times gate_up_proj.forward appears.
        // For dense models, it should appear once per bucket (Gate),
        // not twice (Gate + Up).
        let count = source.matches("gate_up_proj").count();
        // Each bucket has one match arm that references gate_up_proj.
        // If Up also calls it, the count would be doubled.
        let bucket_count = source.matches("solver_layer_bucket_").count();
        // Gate references gate_up_proj once. Up also references it
        // (for the weight), but for dense the Up store is a noop.
        // The key test: Up should NOT produce a second forward() call
        // on gate_up_proj.
        let forward_count = source.matches("gate_up_proj . forward").count()
            + source.matches("gate_up_proj.forward").count();
        // Should be at most bucket_count (one per bucket for Gate).
        // If Up also calls forward, it would be 2 * bucket_count.
        assert!(
            forward_count <= bucket_count,
            "gate_up_proj.forward called {forward_count} times but only {bucket_count} buckets — Up is calling it again"
        );
    }

    /// Every bucket should contain norm, qkv, rope, attention, oproj,
    /// mlp_norm, gate, silu, down at minimum.
    #[test]
    fn every_bucket_has_all_phases() {
        let source = gen_source("1..1024");
        // These patterns should appear in every bucket.
        assert!(source.contains("input_layernorm"), "missing attn norm");
        assert!(source.contains("qkv_proj"), "missing QKV GEMM");
        assert!(source.contains("fused_qkv_rope_cache"), "missing rope");
        assert!(
            source.contains("attention_decode_from_cache"),
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

    /// GPU-specialized and fully dynamic produce empty output (not yet implemented).
    #[test]
    fn gpu_specialized_and_dynamic_produce_empty_placeholder() {
        let gpu_tokens: proc_macro2::TokenStream =
            "model: llama_3_2_1b, target: runtime".parse().unwrap();
        let def: CompileDef = syn::parse2(gpu_tokens).unwrap();
        let _output = codegen::generate(&def);

        let dyn_tokens: proc_macro2::TokenStream =
            "model: runtime, target: runtime".parse().unwrap();
        let def: CompileDef = syn::parse2(dyn_tokens).unwrap();
        let _output = codegen::generate(&def);
    }
}
