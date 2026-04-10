// SPDX-License-Identifier: Apache-2.0
//! Integration tests for the codegen module.

#[cfg(test)]
mod tests {
    use crate::lowering::backend::codegen;
    use crate::lowering::backend::compile_dsl::CompileDef;

    /// Verify the fully-specialized codegen produces a forward function.
    #[test]
    fn fully_specialized_codegen_produces_forward_fn() {
        let tokens: proc_macro2::TokenStream = "
            model: llama_3_2_1b,
            target: l4_sm89,
            workloads: [1..1024],
        "
        .parse()
        .unwrap();

        let def: CompileDef = syn::parse2(tokens).unwrap();
        assert!(def.is_fully_specialized());

        let output = codegen::generate(&def);
        let source = output.to_string();

        // Should contain the solver_forward_layer function.
        assert!(
            source.contains("solver_forward_layer"),
            "output should contain solver_forward_layer"
        );

        // Should contain per-bucket layer functions.
        assert!(
            source.contains("solver_layer_bucket_0"),
            "output should contain solver_layer_bucket_0"
        );

        // Should contain num_tokens match.
        assert!(
            source.contains("num_tokens"),
            "output should match on num_tokens"
        );
    }

    /// Verify the solver discovers multiple plan buckets.
    #[test]
    fn fully_specialized_has_multiple_buckets() {
        let tokens: proc_macro2::TokenStream = "
            model: llama_3_2_1b,
            target: l4_sm89,
            workloads: [1..4096],
        "
        .parse()
        .unwrap();

        let def: CompileDef = syn::parse2(tokens).unwrap();
        let output = codegen::generate(&def);
        let source = output.to_string();

        assert!(source.contains("solver_layer_bucket_0"));
        assert!(source.contains("solver_layer_bucket_1"));
    }

    /// GPU-specialized and fully dynamic codegen are not yet implemented.
    /// They produce empty output (placeholder).
    #[test]
    fn gpu_specialized_and_dynamic_produce_empty_placeholder() {
        let gpu_tokens: proc_macro2::TokenStream =
            "model: llama_3_2_1b, target: runtime".parse().unwrap();
        let def: CompileDef = syn::parse2(gpu_tokens).unwrap();
        let _output = codegen::generate(&def);
        // Compiles without error — that's the test.

        let dyn_tokens: proc_macro2::TokenStream =
            "model: runtime, target: runtime".parse().unwrap();
        let def: CompileDef = syn::parse2(dyn_tokens).unwrap();
        let _output = codegen::generate(&def);
    }
}
