// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Gemma3 — Gemma2 structure plus per-head QK norms and dual RoPE bases.
//!
//! Differences from Gemma2:
//! 1. **Per-head QK norm** — `rmsnorm(q, q_norm + 1.0)` / `rmsnorm(k, k_norm + 1.0)`
//!    between QKV projections and RoPE, same shape as Qwen3's QK-norm
//!    but with Gemma's `(1+w)` offset convention.
//! 2. **Dual RoPE bases** — global-attention layers use `rope_theta`,
//!    sliding-attention layers use `rope_local_base_freq`. The global
//!    rotary lives on `ForwardCtx`; the local one lives on the
//!    compiler-generated `Weights` struct via `rotary_local`.
//! 3. **No softcapping** — neither attention nor final-logit softcap.
//! 4. **5:1 local/global ratio** — `sliding_window_pattern: 6` means
//!    every 6th layer (index 5, 11, 17, …) is global; the rest are
//!    sliding. Predicate: `layer % 6 == 5` → global.

use ferrite_forward::forward;

#[forward(
    workloads = [1, 8, 64, 512, 4096],
)]
fn gemma3() {
    hidden_states = embed(input_ids, embed_tokens) * sqrt(hidden_size);
    for layer in 0..num_hidden_layers {
        pre_attn_normed = rmsnorm(hidden_states, input_layernorm[layer] + 1.0);

        q = gemm(pre_attn_normed, self_attn.q_proj[layer]);
        q = rmsnorm(q, self_attn.q_norm[layer] + 1.0);
        k = gemm(pre_attn_normed, self_attn.k_proj[layer]);
        k = rmsnorm(k, self_attn.k_norm[layer] + 1.0);
        v = gemm(pre_attn_normed, self_attn.v_proj[layer]);

        if layer % sliding_window_pattern == sliding_window_global_remainder {
            (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
            attn = attention(q, k, v, kv_cache[layer], block_table);
        } else {
            (q, k, v) = rope_append(q, k, v, positions, rotary_local, kv_cache[layer]);
            attn = sliding_attention(q, k, v, kv_cache[layer], block_table);
        }

        oproj = gemm(attn, self_attn.o_proj[layer]);
        post_attn_normed = rmsnorm(oproj, post_attention_layernorm[layer] + 1.0);
        hidden_states = add(post_attn_normed, hidden_states);

        pre_ffwd_normed = rmsnorm(hidden_states, pre_feedforward_layernorm[layer] + 1.0);
        gate = gelu(gemm(pre_ffwd_normed, mlp.gate_proj[layer]));
        up = gemm(pre_ffwd_normed, mlp.up_proj[layer]);
        down = gemm(gate * up, mlp.down_proj[layer]);
        post_ffwd_normed = rmsnorm(down, post_feedforward_layernorm[layer] + 1.0);
        hidden_states = add(post_ffwd_normed, hidden_states);
    }
    normed = rmsnorm(hidden_states, norm + 1.0);
    logits = gemm(normed, lm_head);
}
