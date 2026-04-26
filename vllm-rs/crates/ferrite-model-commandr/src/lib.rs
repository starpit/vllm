// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Command R (`CohereForCausalLM`) — the math. Differs from
//! Llama in four places:
//!
//! 1. **CohereLayerNorm** instead of RmsNorm. Subtracts the mean
//!    before scaling: `y = w * (x - mean(x)) / sqrt(var(x) + eps)`.
//!    Weight only, no bias. Surfaced via the new `layer_norm` DSL
//!    op (`OpKind::LayerNorm`).
//! 2. **Parallel attention + MLP**. Both branches read the SAME
//!    pre-norm output and their results are summed back into the
//!    residual: `hidden = hidden + attn(norm(x)) + mlp(norm(x))`.
//!    Only one norm per layer (no `post_attention_layernorm`).
//! 3. **Interleaved RoPE** — pairs adjacent elements `(2i, 2i+1)`
//!    instead of NeoX's `(i, i + half_dim)`. Surfaced via the new
//!    `rope_append_interleaved` DSL op.
//! 4. **Logit scaling** — final logits are multiplied by
//!    `logit_scale` (typically `1/16 = 0.0625`) before sampling.
//!    Expressed as a `scalar(...)` config read on the lm_head
//!    output, which routes through `ScalarMulImpl`.
//!
//! The hand-written reference is `vllm-cuda/src/model/commandr.rs`;
//! `crates/ferrite-model-commandr/configs/weights.json` declares the on-disk
//! tensor names. Probed against `adalbertojunior/c4ai-command-r-v01`
//! (the only confirmed non-gated mirror of the original Cohere
//! arch). `tie_word_embeddings: true` for v01, so the codegen's
//! `LinearTiedToEmbedding` arm wires `lm_head` to the embed buffer.

use ferrite_forward::forward;

#[forward(
    target = "../../../target_profiles/l4_sm89.json",
    workloads = [1, 8, 64, 512, 4096],
)]
fn commandr() {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..num_hidden_layers {
        // One LayerNorm per layer; both attention and MLP read it.
        normed = layer_norm(hidden_states, input_layernorm[layer]);

        // Attention branch.
        q = gemm(normed, self_attn.q_proj[layer]);
        k = gemm(normed, self_attn.k_proj[layer]);
        v = gemm(normed, self_attn.v_proj[layer]);
        (q, k, v) = rope_append_interleaved(q, k, v, positions, rotary, kv_cache[layer]);
        attn = attention(q, k, v, kv_cache[layer], block_table);
        oproj = gemm(attn, self_attn.o_proj[layer]);

        // MLP branch (SwiGLU, same `normed` input).
        gate = silu(gemm(normed, mlp.gate_proj[layer]));
        up = gemm(normed, mlp.up_proj[layer]);
        down = gemm(gate * up, mlp.down_proj[layer]);

        // Three-way residual: hidden += attn; hidden += mlp.
        hidden_states = add(oproj, hidden_states);
        hidden_states = add(down, hidden_states);
    }
    // Final norm + lm_head + logit scale.
    normed = layer_norm(hidden_states, norm);
    logits = gemm(normed, lm_head) * scalar(logit_scale);
}
