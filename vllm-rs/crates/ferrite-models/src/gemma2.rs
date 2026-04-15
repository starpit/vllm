// SPDX-License-Identifier: Apache-2.0
//! Gemma2 — the math. Differs from Llama/Qwen2 in four places:
//!
//! 1. **Four rmsnorms per layer** instead of two. Each sub-block
//!    (attention, MLP) is bracketed by both a pre-norm AND a
//!    post-norm. The residual stream is updated between them.
//! 2. **GELU MLP** (`down(gelu(gate) * up)`) instead of SwiGLU.
//! 3. **Alternating attention** — even layers use full attention,
//!    odd layers use sliding-window attention. The `if layer %
//!    sliding_window_pattern == 0` dispatches per iteration at
//!    unroll time.
//! 4. **Logit soft-cap** — `tanh_softcap` applied to the final
//!    logits.
//!
//! The `(1 + w)` offset on Gemma's rmsnorm weights is a weight-
//! loader concern (applied at load time by the caller), not a DSL
//! or kernel concern; this body just references the rmsnorm ops
//! like any other architecture.
//!
//! The attention softcap and softmax scale (Gemma2 uses
//! `query_pre_attn_scalar.powf(-0.5)` instead of
//! `1/sqrt(head_dim)`) are read from config by the Impl at emit
//! time — nothing Gemma-specific appears here.

use ferrite_forward::forward;

#[forward(
    target = "../../../target_profiles/l4_sm89.json",
    workloads = [1, 8, 64, 512, 4096],
)]
fn gemma2() {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..num_hidden_layers {
        // Pre-attention norm. Gemma's `(1+w)` convention rides as
        // a scalar addition on the weight ref — the solver's
        // ScalarOffsetRmsNormImpl claims the `(Add(w, 1.0), rmsnorm)`
        // pair and emits a single rms_norm call with the 1.0 as
        // the kernel's `weight_offset`.
        pre_attn_normed = rmsnorm(hidden_states, input_layernorm[layer] + 1.0);

        // Attention block.
        q = gemm(pre_attn_normed, self_attn.q_proj[layer]);
        k = gemm(pre_attn_normed, self_attn.k_proj[layer]);
        v = gemm(pre_attn_normed, self_attn.v_proj[layer]);
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
        if layer % sliding_window_pattern == 0 {
            attn = attention(q, k, v, kv_cache[layer], block_table);
        } else {
            attn = sliding_attention(q, k, v, kv_cache[layer], block_table);
        }
        oproj = gemm(attn, self_attn.o_proj[layer]);

        // Post-attention norm on the attention output, before residual.
        post_attn_normed = rmsnorm(oproj, post_attention_layernorm[layer] + 1.0);

        // Residual + pre-ffwd norm.
        hidden_states = add(post_attn_normed, hidden_states);
        pre_ffwd_normed = rmsnorm(hidden_states, pre_feedforward_layernorm[layer] + 1.0);

        // GELU MLP — solver claims `(gemm, gemm, gelu, mul)` as a
        // FusedGateUpGeluMul subgraph.
        gate = gelu(gemm(pre_ffwd_normed, mlp.gate_proj[layer]));
        up = gemm(pre_ffwd_normed, mlp.up_proj[layer]);
        down = gemm(gate * up, mlp.down_proj[layer]);

        // Post-ffwd norm on the MLP output, before residual.
        post_ffwd_normed = rmsnorm(down, post_feedforward_layernorm[layer] + 1.0);

        hidden_states = add(post_ffwd_normed, hidden_states);
    }
    // Final norm + lm_head + logit softcap.
    normed = rmsnorm(hidden_states, norm + 1.0);
    logits = gemm(normed, lm_head);
    capped = tanh_softcap(logits);
}
