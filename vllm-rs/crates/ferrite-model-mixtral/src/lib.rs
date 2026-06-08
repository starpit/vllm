// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Mixtral (`MixtralForCausalLM`) — sparse MoE decoder.
//!
//! Structurally identical to Llama / Mistral except the per-layer MLP
//! is replaced by a fused MoE block (`block_sparse_moe`). 8 experts,
//! top-2 softmax routing, no shared expert, no renormalization. The
//! MoE weight on disk lives at
//! `model.layers.{l}.block_sparse_moe.{gate,experts.{e}.{w1,w2,w3}}`;
//! the per-layer accessor is `block_sparse_moe[layer]` returning a
//! `FusedMoELayer`.
//!
//! Reference: `vllm-cuda/src/model/mixtral.rs`. The hand-written model
//! supports BF16 / FP8-dynamic / Marlin (AWQ/GPTQ) — ferrite ships the
//! BF16 path here; quant variants land alongside their dedicated
//! Impls in a follow-up.

use ferrite_forward_macro::forward;

#[forward(
    sk_buckets = [128, 512, 2048, 8192],
)]
fn mixtral() {
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
        mlp_out = moe_block(normed2, block_sparse_moe[layer]);
        hidden_states = add(mlp_out, hidden_states);
    }
    normed = rmsnorm(hidden_states, norm);
    logits = gemm(normed, lm_head);
}
