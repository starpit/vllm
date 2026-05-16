// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Phi-3 (`Phi3ForCausalLM`) — structurally identical to Llama
//! (RMSNorm, SwiGLU MLP, RoPE-NeoX, no projection biases). The body
//! below is a verbatim copy of `mistral.rs` / `llama.rs`.
//!
//! Scope: `microsoft/Phi-3-mini-4k-instruct` only. This variant is
//! dense bf16, MHA (`num_attention_heads == num_key_value_heads`),
//! and has `rope_scaling: null` (`max_position_embeddings=4096`) —
//! no LongRoPE plumbing needed. `sliding_window=2047` is present in
//! the config but is inert at `max_model_len <= 2047`; we do not
//! wire a sliding-window DSL primitive here (same stance as Mistral
//! v0.1). LongRoPE / GQA variants (Phi-3.5-mini-128k, Phi-4-mini)
//! are explicit out-of-scope — they share this arch string
//! (`Phi3ForCausalLM`) but need new RoPE-scaling bound-gated
//! machinery and, for Phi-4, GQA-aware packed-qkv splits.
//!
//! Packed weights: Phi-3 ships one `self_attn.qkv_proj.weight`
//! (`[3*hidden, hidden]` under MHA) and one `mlp.gate_up_proj.weight`
//! (`[2*intermediate, hidden]`) per layer instead of the five logical
//! tensors the DSL body references. The per-slice reads below resolve
//! via a packed-source fallback in `ferrite_kernels::layers::Linear::load`
//! — see that function for the slicing logic. The DSL body itself is
//! unaware of the packing; it asks for `q_proj` / `k_proj` / `v_proj`
//! / `gate_proj` / `up_proj` by name like every other llama-shaped
//! arch.
//!
//! `tie_word_embeddings=false` for Phi-3-mini-4k, so the loader reads
//! a real `lm_head.weight` — same as Mistral.

use ferrite_forward_macro::forward;

#[forward(
    workloads = [1, 8, 64, 512, 4096],
    sk_buckets = [128, 512, 2048, 8192],
)]
fn phi3() {
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
