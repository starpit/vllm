// Temporary test to dump generated source for debugging.
#[cfg(test)]
mod tests {
    use crate::lowering::backend::codegen;
    use crate::lowering::backend::compile_dsl::ForwardDef;

    #[test]
    fn dump_bucket_0_source() {
        let tokens: proc_macro2::TokenStream = r#"
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..NL {
                let normed = rmsnorm(hidden_states, input_layernorm[layer]);
                let q = gemm(normed, self_attn.q_proj[layer]);
                let k = gemm(normed, self_attn.k_proj[layer]);
                let v = gemm(normed, self_attn.v_proj[layer]);
                let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
                hidden_states = gemm_add(attn, self_attn.o_proj[layer], hidden_states);

                let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                let up = gemm(normed2, mlp.up_proj[layer]);
                hidden_states = gemm_add(gate * up, mlp.down_proj[layer], hidden_states);
            }
            hidden_states = rmsnorm(hidden_states, norm);
            logits = gemm(hidden_states, lm_head);

            models: [
                { layers: 16, hidden: 2048, intermediate: 8192, heads: 32, kv_heads: 8, head_dim: 64 },
            ],
            target: l4_sm89,
            workloads: [1..4096],
        "#
        .parse()
        .unwrap();
        let def: ForwardDef = syn::parse2(tokens).unwrap();
        let output = codegen::generate(&def);
        let source = output.to_string();
        eprintln!("\n=== GENERATED SOURCE ===\n{source}\n=== END ===\n");
    }
}
