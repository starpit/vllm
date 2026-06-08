// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::possible_missing_comma)]
//! DeepSeek V3 (`DeepseekV3ForCausalLM`) — MLA + sigmoid-gated MoE decoder.
//!
//! Differences from DeepSeek V2 Lite (`ferrite-model-deepseek-v2`):
//! 1. **Q lora-rank path**: Q projection is split into
//!    `q_a_proj → q_a_layernorm → q_b_proj` instead of a single `q_proj`.
//! 2. **Sigmoid MoE routing** (`scoring_func="sigmoid"`, `topk_method="noaux_tc"`):
//!    experts selected by `sigmoid(logit) + e_score_correction_bias` (biased),
//!    weights are unbiased `sigmoid(logit)`, optionally renormalized.
//! 3. All other components (KV path, MLA attention, dense layer 0, shared expert)
//!    are identical to V2.
//!
//! Reference: Python vLLM `vllm/model_executor/models/deepseek_v2.py`
//!   — `DeepseekV3ForCausalLM` is an empty subclass of `DeepseekV2ForCausalLM`.

use ferrite_forward_macro::forward;

#[forward()]
mod deepseek_v3 {
    // DeepSeek's `moe` DSL weight name maps to `mlp` on disk (HF
    // safetensors store the MoE block under `model.layers.{l}.mlp`).
    const WEIGHT_LEAF_RENAMES: &[(&str, &str)] = &[("moe", "mlp")];

    fn forward() {
        hidden_states = embed(input_ids, embed_tokens);
        for layer in 0..num_hidden_layers {
            // ── Attention ──────────────────────────────────────────────
            normed = rmsnorm(hidden_states, input_layernorm[layer]);

            // Q path: q_a_proj → q_a_layernorm → q_b_proj (q_lora_rank path)
            q_a = gemm(normed, self_attn.q_a_proj[layer]);
            q_a = rmsnorm(q_a, self_attn.q_a_layernorm[layer]);
            q = gemm(q_a, self_attn.q_b_proj[layer]);

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
                // Dense SwiGLU MLP (layer 0 only).
                mlp_out = gemm(
                    silu(gemm(normed2, mlp.gate_proj[layer])) * gemm(normed2, mlp.up_proj[layer]),
                    mlp.down_proj[layer],
                );
            } else {
                // DeepSeek MoE: sigmoid-routed experts + shared expert
                mlp_out = moe_block(normed2, moe[layer]);
            }
            hidden_states = add(mlp_out, hidden_states);
        }
        normed = rmsnorm(hidden_states, norm);
        logits = gemm(normed, lm_head);
    }
}
