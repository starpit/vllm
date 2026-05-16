// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Qwen3 — Llama math plus per-head RMS norm on Q and K (between the
//! QKV projections and RoPE). No QKV bias (unlike Qwen2). The per-head
//! norm weights `self_attn.q_norm` / `self_attn.k_norm` are declared
//! in `crates/ferrite-model-qwen3/configs/weights.json` at shape `[head_dim]`;
//! shape inference sees the inferred `[T, heads*head_dim]` activation
//! vs. the declared `[head_dim]` weight and synthesizes the view
//! reshape tiles to bridge them.

use ferrite_forward_macro::forward;

#[forward(
    workloads = [1, 8, 64, 512, 4096],
)]
fn qwen3() {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..num_hidden_layers {
        normed = rmsnorm(hidden_states, input_layernorm[layer]);
        q = gemm(normed, self_attn.q_proj[layer]);
        q = rmsnorm(q, self_attn.q_norm[layer]);
        k = gemm(normed, self_attn.k_proj[layer]);
        k = rmsnorm(k, self_attn.k_norm[layer]);
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
