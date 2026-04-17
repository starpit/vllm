// SPDX-License-Identifier: Apache-2.0
//! Granite (IBM) — Llama's math body plus four scalar multipliers
//! read from config:
//!
//!   1. `embedding_multiplier` — scales the embed output.
//!   2. `residual_multiplier`  — scales the attention and MLP outputs
//!      before each residual add.
//!   3. `attention_multiplier` — direct softmax scale (NOT subject
//!      to Gemma's `.powf(-0.5)` transform). Read inside attention
//!      impls via `attention_scale_for`; nothing DSL-level here.
//!   4. `logits_scaling`       — logits are divided by this scalar
//!      post-lm_head, expressed in the DSL as
//!      `logits * recip_scalar(logits_scaling)`.
//!
//! All four values come from top-level numeric fields of the model's
//! `config.json` and ride into the FUF via `scalar(<name>)` /
//! `recip_scalar(<name>)`, which `cfg.rs::fold_scalars` resolves to
//! `ScalarLit` at CFG-build time.

use ferrite_forward::forward;

#[forward(
    target = "../../../target_profiles/l4_sm89.json",
    workloads = [1, 8, 64, 512, 4096],
    sk_buckets = [128, 512, 2048, 8192],
)]
fn granite() {
    hidden_states = embed(input_ids, embed_tokens) * scalar(embedding_multiplier);
    for layer in 0..num_hidden_layers {
        normed = rmsnorm(hidden_states, input_layernorm[layer]);
        q = gemm(normed, self_attn.q_proj[layer]);
        k = gemm(normed, self_attn.k_proj[layer]);
        v = gemm(normed, self_attn.v_proj[layer]);
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
        attn = attention(q, k, v, kv_cache[layer], block_table);
        oproj = gemm(attn, self_attn.o_proj[layer]);
        hidden_states = add(oproj * scalar(residual_multiplier), hidden_states);

        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
        gate = silu(gemm(normed2, mlp.gate_proj[layer]));
        up = gemm(normed2, mlp.up_proj[layer]);
        down = gemm(gate * up, mlp.down_proj[layer]);
        hidden_states = add(down * scalar(residual_multiplier), hidden_states);
    }
    normed = rmsnorm(hidden_states, norm);
    logits_raw = gemm(normed, lm_head);
    logits = logits_raw * recip_scalar(logits_scaling);
}
