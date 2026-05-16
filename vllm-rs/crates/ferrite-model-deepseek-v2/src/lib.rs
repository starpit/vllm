// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::possible_missing_comma)]
//! DeepSeek V2 Lite (`DeepseekV2ForCausalLM`) — MLA + MoE decoder.
//!
//! Architecture highlights:
//! 1. **MLA (Multi-Latent Attention)**: For V2-Lite, `q_lora_rank=null` so
//!    the Q path is a single `q_proj` directly to `[T, heads*qk_head_dim]`.
//!    KV path: kv_a_proj_with_mqa → mla_split → kv_a_layernorm → kv_b_proj.
//!    Full K/V tensors assembled in-flight; writes to standard paged cache.
//! 2. **DeepSeek MoE**: routed experts × routed_scaling_factor + shared expert
//!    (plain ADD — no sigmoid gate unlike Qwen2/3 MoE).
//! 3. **Dense first layer**: Layer 0 uses standard SwiGLU MLP; layers 1+ use MoE.
//! 4. **Interleaved RoPE** (YaRN) on the rope portion of Q and k_pe.
//!
//! Reference: `vllm-cuda/src/model/deepseek_v2.rs`.

use ferrite_forward_macro::forward;

#[forward(
    workloads = [1, 8, 64, 512, 4096],
)]
fn deepseek_v2() {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..num_hidden_layers {
        // ── Attention ──────────────────────────────────────────────
        normed = rmsnorm(hidden_states, input_layernorm[layer]);

        // Q path: direct q_proj (q_lora_rank=null for V2-Lite)
        q = gemm(normed, self_attn.q_proj[layer]);

        // KV path: kv_a_proj_with_mqa → mla_split → kv_a_layernorm → kv_b_proj
        kv_a = gemm(normed, self_attn.kv_a_proj_with_mqa[layer]);
        (kv_latent, k_pe) = mla_split(kv_a);
        kv_latent = rmsnorm(kv_latent, self_attn.kv_a_layernorm[layer]);
        kv_b = gemm(kv_latent, self_attn.kv_b_proj[layer]);

        // MLA attention: assembles K/V, applies interleaved RoPE, writes cache
        attn = mla_attention(
            q,
            kv_b,
            k_pe,
            positions,
            rotary,
            kv_cache[layer],
            block_table,
        );
        oproj = gemm(attn, self_attn.o_proj[layer]);
        hidden_states = add(oproj, hidden_states);

        // ── MLP / MoE ──────────────────────────────────────────────
        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
        if layer < first_k_dense_replace {
            // Dense SwiGLU MLP (layer 0 only for V2-Lite).
            // Inlined into one expression so both branches bind only `mlp_out`.
            mlp_out = gemm(
                silu(gemm(normed2, mlp.gate_proj[layer])) * gemm(normed2, mlp.up_proj[layer]),
                mlp.down_proj[layer],
            );
        } else {
            // DeepSeek MoE: routed experts + shared expert
            mlp_out = moe_block(normed2, moe[layer]);
        }
        hidden_states = add(mlp_out, hidden_states);
    }
    normed = rmsnorm(hidden_states, norm);
    logits = gemm(normed, lm_head);
}
