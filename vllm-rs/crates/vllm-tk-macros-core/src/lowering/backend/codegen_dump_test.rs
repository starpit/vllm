// Temporary test to dump generated source for debugging.
#[cfg(test)]
mod tests {
    use crate::lowering::backend::codegen;
    use crate::lowering::backend::compile_dsl::CompileDef;

    #[test]
    fn dump_bucket_0_source() {
        let tokens: proc_macro2::TokenStream =
            "model: llama_3_2_1b, target: l4_sm89, workloads: [1..4096]"
                .parse()
                .unwrap();
        let def: CompileDef = syn::parse2(tokens).unwrap();
        let output = codegen::generate(&def);
        let source = output.to_string();

        // Pretty-print via prettyplease if available, else raw.
        eprintln!("\n=== GENERATED SOURCE ===\n{source}\n=== END ===\n");
    }
}
