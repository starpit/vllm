// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Mistral — structurally identical to Llama (RMSNorm, SwiGLU MLP,
//! GQA, RoPE, no projection biases). The body below is a verbatim
//! copy of `llama.rs`.
//!
//! Per-model differences that do not show up here:
//! - `rope_theta` is read from each `<size>.json` at macro-expansion
//!   and baked into the emitted `RotaryCache` — body is
//!   frequency-agnostic (Mistral 7B v0.3/Nemo use 1e6, not 1e4).
//! - Mistral-Nemo breaks the `hidden_size == num_attention_heads *
//!   head_dim` coincidence (`5120 != 32 * 128`). The weights.json
//!   manifest encodes `head_dim * num_attention_heads` on q_proj and
//!   o_proj so shape inference is correct on both 7B and Nemo.
//!
//! Scope: ships only the non-sliding variants (v0.2, v0.3, Nemo —
//! every `TestModels::MISTRAL` target). Mistral-7B-v0.1 and Zephyr
//! (both `sliding_window: 4096` applied on every layer) are
//! intentionally not covered — they would need either a new
//! bound-gated sliding primitive or a split arch directory, which
//! has no consumer yet.
//!
//! `tie_word_embeddings=false` across every Mistral size, so the
//! loader reads a real `lm_head.weight` from safetensors; same as
//! Llama's untied variants.

use ferrite_forward::forward;

#[forward(
    target = "../../../target_profiles/l4_sm89.json",
    workloads = [1, 8, 64, 512, 4096],
    sk_buckets = [128, 512, 2048, 8192],
)]
fn mistral() {
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
