// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::possible_missing_comma)]
//! DeepSeek V3 (`DeepseekV3ForCausalLM`), **flat-Q variant** —
//! `q_lora_rank=null` so the Q path is a single `self_attn.q_proj`
//! (V2-style direct), but the routing flavor is V3-style sigmoid+
//! noaux_tc with grouped top-k.
//!
//! This is `ferrite-model-deepseek-v3`'s sibling for checkpoints
//! that ship `DeepseekV3ForCausalLM` arch but use V2-Lite's direct
//! Q projection instead of V3's `q_a_proj → q_a_layernorm → q_b_proj`.
//! Both crates register the same HF arch identifier; cross-arch
//! dispatch falls through on `Ok(None)` so the right one's
//! fingerprint wins per-checkpoint.
//!
//! Covered today: `moonshotai/Moonlight-16B-A3B-Instruct` — 16 GB
//! BF16, fits L4. The only `DeepseekV3ForCausalLM` + flat-Q +
//! K2-style routing fixture small enough to validate end-to-end on
//! commodity hardware (real Kimi-K2 / K2.5 / K2.6 are 1T-scale).
//!
//! Routing flavor is config-driven through the `moe_block(..)` op:
//! `scoring_func="sigmoid"` + `topk_method="noaux_tc"` flips
//! `use_sigmoid` in the loaded `DeepSeekV2MoELayer`, and `n_group` /
//! `topk_group` / `routed_scaling_factor` thread through unchanged.
//! Same DSL body as `ferrite-model-deepseek-v2`; only the configs
//! and the registered HF arch differ.

use ferrite_forward_macro::forward;

#[forward(
    workloads = [1, 8, 64, 512, 4096],
)]
fn deepseek_v3_flat() {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..num_hidden_layers {
        // ── Attention ──────────────────────────────────────────────
        normed = rmsnorm(hidden_states, input_layernorm[layer]);

        // Q path: direct q_proj (q_lora_rank=null).
        q = gemm(normed, self_attn.q_proj[layer]);

        // KV path: kv_a_proj_with_mqa → mla_split → kv_a_layernorm → kv_b_proj.
        kv_a = gemm(normed, self_attn.kv_a_proj_with_mqa[layer]);
        (kv_latent, k_pe) = mla_split(kv_a);
        kv_latent = rmsnorm(kv_latent, self_attn.kv_a_layernorm[layer]);
        kv_b = gemm(kv_latent, self_attn.kv_b_proj[layer]);

        // MLA attention: assembles K/V, applies interleaved RoPE, writes cache.
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
            // Dense SwiGLU MLP for the first `first_k_dense_replace` layers.
            mlp_out = gemm(
                silu(gemm(normed2, mlp.gate_proj[layer])) * gemm(normed2, mlp.up_proj[layer]),
                mlp.down_proj[layer],
            );
        } else {
            // DeepSeek MoE — sigmoid+noaux_tc routing chosen per-config.
            mlp_out = moe_block(normed2, moe[layer]);
        }
        hidden_states = add(mlp_out, hidden_states);
    }
    normed = rmsnorm(hidden_states, norm);
    logits = gemm(normed, lm_head);
}
