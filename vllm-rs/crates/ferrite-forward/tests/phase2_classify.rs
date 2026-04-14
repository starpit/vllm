// SPDX-License-Identifier: Apache-2.0
//! Phase 2 integration: classification errors surface as compile
//! errors at the macro call site. Observing-the-claim test: if the
//! classifier regresses, the real `#[forward]` macro path breaks.
//!
//! Trybuild would be the ideal harness for negative tests (checking
//! compile-failures), but we can also observe classifier success
//! through the positive path — the realistic Llama body from Phase 1
//! must compile through parse + classify.

use ferrite_forward::forward;

// Reuses the full Llama body from Phase 1. If the classifier rejects
// any reference in the body (extern param, weight ref, local
// binding, loop-indexed access), this test fails to compile.
#[forward]
fn llama_full() {
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
}

#[test]
fn classified_fn_is_callable() {
    llama_full();
}
