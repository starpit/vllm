// SPDX-License-Identifier: Apache-2.0
//! Qwen2 / Qwen2.5 — same math as Llama, differs only in that the
//! QKV projections carry bias. Ferrite's fused-QKV weight accessor
//! uses `LinearLayer::load_dense_concat` which auto-detects biases
//! on the source weights and packs them into the fused `LinearLayer`;
//! `Linear::forward` then fuses the bias add into cuBLAS's GEMM epilog
//! via `gemm_bias`. No DSL-level `bias_add` op is needed.
//!
//! One `#[forward]` body per architecture; per-model configs fan out
//! via `model_architectures/qwen2/*.json`.

use ferrite_forward::forward;

#[forward(
    target = "../../../target_profiles/l4_sm89.json",
    workloads = [1, 8, 64, 512, 4096],
)]
fn qwen2() {
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
